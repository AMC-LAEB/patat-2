import rasterio
import numpy as np
import scipy.stats as st
from scipy.optimize import linprog
import pandas as pd
import re
from numba import jit, prange, vectorize
from numba import float32, int32

#@jit(nopython=True, parallel=True, fastmath=True)
#def simulate_surveillance(symp_inds, symp_days, people_locadd, populated_grid_to_hcf_dist, ndays):



@jit(nopython=True, parallel=True, fastmath=True)
def identify_rand_sus(infectious_locadd, susceptible_locadd, susceptible_inds, commute_kernel, ind_commute_kernel_sum):
    n_sus = susceptible_locadd.size
    sus_rand_mask = np.zeros(n_sus, dtype=np.int8)
    sus_meetinf_lamb = np.zeros(n_sus, dtype=np.float32)
    # for each susceptible individual
    for s in prange(n_sus):
        suslocadd = susceptible_locadd[s]
        susind = susceptible_inds[s]
        # will susceptible meet infectious person
        suskernel = commute_kernel[suslocadd]
        meetinf_lamb = suskernel[infectious_locadd]/ind_commute_kernel_sum[susind]
        meetinf_prob = 1 - np.exp(-meetinf_lamb.sum())
        if np.random.random() < meetinf_prob:
            sus_rand_mask[s] = 1
    return susceptible_inds[sus_rand_mask>0]

@jit(nopython=True, parallel=True, fastmath=True)
def compute_maxdist_coverage(i_loc, latlon, max_dist_travelled):
    n = i_loc.shape[0] # n populated locations
    norms = np.zeros((n, n), dtype=np.float32)
    for i in prange(n):
        icoords = i_loc[i]
        ilatlon = latlon[icoords[0], icoords[1]]
        j_latlon = slice3d_with_2d(latlon, i_loc)
        d = haversine_d(ilatlon, j_latlon)
        kernel = np.zeros(n, dtype=np.int8)
        kernel[d<=max_dist_travelled] = 1
        norms[i,:] = kernel
    return norms

@jit(nopython=True, parallel=True, fastmath=True)
def compute_commute_kernel(i_loc, latlon, people_locadd, a, b):
    n = i_loc.shape[0] # n populated locations
    n_people = people_locadd.size
    norms = np.zeros((n, n), dtype=np.float32)
    norms_sum = np.zeros(n_people, dtype=np.float32)
    for i in prange(n):
        icoords = i_loc[i]
        ilatlon = latlon[icoords[0], icoords[1]]
        j_latlon = slice3d_with_2d(latlon, i_loc)
        d = haversine_d(ilatlon, j_latlon)
        # compute commute kernel
        kernel = 1 / (1 + (d/a) ** b)
        norms_sum[people_locadd == i] = kernel[people_locadd].sum()
        norms[i,:] = kernel
    return norms, norms_sum

@jit(nopython=True, parallel=True, fastmath=True)
def compute_rand_transmission(infectious_inds, infectious_locadd, overdispersion_f, isolate_red_f, susceptible_locadd, relsus_f, protection_from_prev_immunity_p, beta, setting_beta_f, contact_mat, commute_kernel, ind_commute_kernel_sum, infectious_agebin, susceptible_agebin, susceptible_inds, infectious_vars, susceptible_states, mt_trans_f, cross_wt_immunity):
    # commute_kernel, ind_commute_kernel_sum,
    n_sus = susceptible_locadd.size
    sus_infected_mask = np.zeros(n_sus, dtype=np.int8)
    sus_infector = np.zeros(n_sus, dtype=np.int32)

    # for each susceptible individual
    for s in prange(n_sus):
        suslocadd = susceptible_locadd[s]
        susind = susceptible_inds[s]
        # will susceptible meet infectious person
        suskernel = commute_kernel[suslocadd]
        meetinf_lamb = suskernel[infectious_locadd]/ind_commute_kernel_sum[susind]
        # get mean contact rates
        susagebin = susceptible_agebin[s]
        suscontact = contact_mat[:,susagebin]
        mean_contact = suscontact[infectious_agebin]
        # compute poisson lamb
        lamb = setting_beta_f * beta * mean_contact * meetinf_lamb * relsus_f[s] * protection_from_prev_immunity_p[s] * overdispersion_f * isolate_red_f

        wt_lamb = lamb[infectious_vars == 0]
        mt_lamb = lamb[infectious_vars > 0] * mt_trans_f

        susstate = susceptible_states[s]
        if susstate > 0:
            # adjust lamb if susceptible was infected by wt before
            wt_lamb_sum = 0.
            mt_lamb_sum = mt_lamb.sum() * (1. - cross_wt_immunity)
        else:
            wt_lamb_sum = wt_lamb.sum()
            mt_lamb_sum = mt_lamb.sum()

        # compute probability for time step = 1
        wt_trans_prob = 1 - np.exp(-wt_lamb_sum)
        mt_trans_prob = 1 - np.exp(-mt_lamb_sum)
        total_trans_prob = wt_trans_prob + mt_trans_prob

        # generate random float
        diethrow = np.random.random()
        if diethrow < total_trans_prob:
            if diethrow < wt_trans_prob:
                sus_infected_mask[s] = 1
                # pick an infector
                wt_infectors = infectious_inds[infectious_vars == 0]
                cumm_p_arr = np.cumsum(wt_lamb/wt_lamb_sum)
                sus_infector[s] = wt_infectors[np.searchsorted(cumm_p_arr, np.random.random(), side="right")]
            else:
                sus_infected_mask[s] = 2
                # pick an infector
                mt_infectors = infectious_inds[infectious_vars > 0]
                cumm_p_arr = np.cumsum(mt_lamb/mt_lamb_sum)
                sus_infector[s] = mt_infectors[np.searchsorted(cumm_p_arr, np.random.random(), side="right")]

    print (sus_infected_mask[sus_infected_mask==1].size, sus_infected_mask[sus_infected_mask==2].size, 'rand')
    return sus_infected_mask, sus_infector

@jit(nopython=True, parallel=True, fastmath=True)
def compute_transmission(infectious_inds, infectious_places, overdispersion_f, isolate_red_f, susceptible_places, relsus_f, protection_from_prev_immunity_p, beta, setting_beta_f, contact_mat, infectious_agebin, susceptible_agebin, infectious_vars, susceptible_states, mt_trans_f, cross_wt_immunity):
    n_sus = susceptible_places.size
    sus_infected_mask = np.zeros(n_sus, dtype=np.int8)
    sus_infector = np.zeros(n_sus, dtype=np.int32)

    # for each susceptible individual
    for s in prange(n_sus):
        susplace = susceptible_places[s]
        susagebin = susceptible_agebin[s]
        suscontact = contact_mat[:,susagebin]

        infplace_mask = infectious_places == susplace
        infectors = infectious_inds[infplace_mask]
        mean_contact = suscontact[infectious_agebin[infplace_mask]]

        # compute poisson lambda
        lamb = setting_beta_f * beta * mean_contact * relsus_f[s] * protection_from_prev_immunity_p[s] * overdispersion_f[infplace_mask] * isolate_red_f[infplace_mask]

        wt_lamb = lamb[infectious_vars[infplace_mask] == 0]
        mt_lamb = lamb[infectious_vars[infplace_mask] > 0] * mt_trans_f

        susstate = susceptible_states[s]
        if susstate > 0:
            # adjust lamb if susceptible was infected by wt before
            wt_lamb_sum = 0.
            mt_lamb_sum = mt_lamb.sum() * (1. - cross_wt_immunity)
        else:
            wt_lamb_sum = wt_lamb.sum()
            mt_lamb_sum = mt_lamb.sum()

        # compute probability for time step = 1
        wt_trans_prob = 1 - np.exp(-wt_lamb_sum)
        mt_trans_prob = 1 - np.exp(-mt_lamb_sum)
        total_trans_prob = wt_trans_prob + mt_trans_prob

        # generate random float
        diethrow = np.random.random()
        if diethrow < total_trans_prob:
            if diethrow < wt_trans_prob:
                sus_infected_mask[s] = 1
                # pick an infector
                wt_infectors = infectors[infectious_vars[infplace_mask] == 0]
                cumm_p_arr = np.cumsum(wt_lamb/wt_lamb_sum)
                sus_infector[s] = wt_infectors[np.searchsorted(cumm_p_arr, np.random.random(), side="right")]
            else:
                sus_infected_mask[s] = 2
                # pick an infector
                mt_infectors = infectors[infectious_vars[infplace_mask] > 0]
                cumm_p_arr = np.cumsum(mt_lamb/mt_lamb_sum)
                sus_infector[s] = mt_infectors[np.searchsorted(cumm_p_arr, np.random.random(), side="right")]

    print (n_sus, sus_infected_mask[sus_infected_mask==1].size, sus_infected_mask[sus_infected_mask==2].size)
    return sus_infected_mask, sus_infector

@jit(nopython=True)
def update_populated_sus_locadd(populated_sus_locadd, exposed_locadd, exposed_agebin):
    for i in np.arange(exposed_locadd.size):
        locadd = exposed_locadd[i]
        agebin = exposed_agebin[i]
        if populated_sus_locadd[locadd, agebin] < 0:
            raise Exception("UGH!")
        populated_sus_locadd[locadd, agebin] -= 1
    return populated_sus_locadd

@jit(nopython=True)
def get_age_dist_of_grids(populated_grid_locadd, people_locadd, people_agebin):
    locadd_agedist = np.zeros((populated_grid_locadd.size, 20), dtype=np.int32)
    for p in np.arange(people_locadd.size):
        loc = people_locadd[p]
        agebin = people_agebin[p]
        locadd_agedist[loc, agebin] += 1
    return locadd_agedist

def get_others_contact_mat(datadir, country_iso3, loc_type):
    contact_rates_df = pd.read_csv(datadir + "prem-et-al_synthetic_contacts_2020.csv")
    contact_rates_df['iso3c'] = contact_rates_df['iso3c'].str.lower()
    # other
    contact_rates_df = contact_rates_df[(contact_rates_df['setting']=='overall')&(contact_rates_df['location_contact']==loc_type)]
    contact_rates_df = contact_rates_df.set_index(['iso3c', 'age_contactor', 'age_cotactee']).sort_index()
    contact_rates_mat = contact_rates_df.loc[country_iso3.lower()]['mean_number_of_contacts'].reset_index().pivot(index='age_contactor', columns='age_cotactee', values='mean_number_of_contacts').to_numpy()
    return contact_rates_mat

def compute_latent_infectious_trajectories(start_t, peak_t_arr, end_t_arr, start_vload, peak_vload, inf_vload):
    # array-ize start_t
    start_t_arr = np.zeros(peak_t_arr.size, dtype=np.int32) + start_t
    # get latent period
    latent_tau = start_t + np.around((abs(inf_vload - start_vload)/(abs(peak_vload - start_vload)/(peak_t_arr-start_t_arr)))).astype(np.int32)
    # get infectious period
    infectious_tau = latent_tau + np.around((abs(peak_vload - inf_vload)/(abs(peak_vload - start_vload)/(end_t_arr-peak_t_arr)))).astype(np.int32)
    return latent_tau, infectious_tau

def rand_generator(par_dict, ind_agebin, minval=1, vtype=np.int32):
    if par_dict['dist'] == 'point':
        return par_dict['par1'][ind_agebin].astype(vtype)

    elif par_dict['dist'] == 'uniform':
        return np.maximum(minval, np.atleast_1d(st.uniform.rvs(loc=par_dict['par1'][ind_agebin], scale=par_dict['par2'][ind_agebin]-par_dict['par1'][ind_agebin]))).astype(vtype)

    elif par_dict['dist'] == 'lognormal':
        return np.maximum(minval, np.atleast_1d(np.log(st.lognorm.rvs(s=par_dict['par1'][ind_agebin], scale=np.exp(par_dict['par2'][ind_agebin]))))).astype(vtype)

def negbin(mean, shape, size, step = 0.1):
    nbn_n = shape
    nbn_p = shape/(mean/step + shape)
    return np.atleast_1d(st.nbinom.rvs(n=nbn_n, p=nbn_p, size=size) * step).astype(np.float32)

@jit(nopython=True)
def update_arr_counts(arr, update):
    for i in update:
        if i > - 1:
            arr[i] += 1
    return arr

@jit(nopython=True)
def workplace_choose_ind(iid_arr, i_arr, j_arr, n_arr, max_km):

    j_idx_arr = np.arange(j_arr.shape[0], dtype=np.int32)
    sel_j_arr = np.zeros(j_arr.shape[0], dtype=np.int32) - 1

    for i in np.arange(i_arr.shape[0], dtype=np.int32):
        if sel_j_arr[sel_j_arr<0].size == 0:
            break
        # get coordinates of i
        ix, iy = i_arr[i,:]
        # get j indices withint max_km of i
        min_ix = np.maximum(0, ix - max_km)
        max_ix = np.minimum(j_arr[:,0].max(), ix + max_km)
        min_iy = np.maximum(0, iy - max_km)
        max_iy = np.minimum(j_arr[:,1].max(), iy + max_km)
        shortlist_j = j_idx_arr[(j_arr[:,0]>=min_ix)&(j_arr[:,0]<=max_ix)&(j_arr[:,1]>=min_iy)&(j_arr[:,1]<=max_iy)&(sel_j_arr<0)]
        # get n
        n = n_arr[i]
        if shortlist_j.size == 0:
            continue
        elif shortlist_j.size > n:
            # randomly select n number of j
            sel_j = np.random.choice(shortlist_j, n, replace=False)
        else:
            sel_j = shortlist_j[:]
        sel_j_arr[sel_j] = iid_arr[i]
    return sel_j_arr

@jit(nopython=True)
def haversine_d(latlon1, latlon2, R=6371):
    lat1, lon1 = np.radians(latlon1)
    latlon2 = np.radians(latlon2)
    lat2 = latlon2[:,0]
    lon2 = latlon2[:,1]
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

@jit(nopython=True)
def slice3d_with_2d(arr3d, arr2d):
    new_arr = np.zeros((arr2d.shape[0], 2), dtype=arr3d.dtype)
    for i in prange(arr2d.shape[0]):
        new_arr[i,:] = arr3d[arr2d[i,0], arr2d[i,1]]
    return new_arr

@jit(nopython=True)
def compute_norms(i_loc, j_loc, latlon):
    norms = np.zeros((i_loc.shape[0], j_loc.shape[0]), dtype=np.float32)
    for i in prange(i_loc.shape[0]):
        icoords = i_loc[i,:]
        # numba slicing workaround
        j_latlon = slice3d_with_2d(latlon, j_loc)
        # compute by haversine distance
        norms[i,:] = haversine_d(latlon[icoords[0], icoords[1]], j_latlon)
    return norms

@jit(nopython=True)
def choose_nearest_idx_by_softmax(probs, n):
    # compute probabilities by softmax
    p_arr = np.exp(probs)/np.sum(np.exp(probs))
    # compute cummulative p
    curr_p_arr = p_arr.copy()
    curr_p_arr = curr_p_arr/curr_p_arr.sum()
    cumm_p_arr = np.cumsum(curr_p_arr)
    # randomly select n
    selected_mask = np.zeros(probs.size, dtype=np.int8)
    select_idx = np.unique(np.searchsorted(cumm_p_arr, np.random.random(n), side="right")).astype(np.int32)
    selected_mask[select_idx] = 1
    curr_size = selected_mask.sum()
    # chec if n == curr_size
    while n - curr_size > 0:
        add_n = n - curr_size
        # recompute cumm. p without those that had been selected
        curr_p_arr = p_arr[selected_mask==0].copy()
        curr_p_arr = curr_p_arr/curr_p_arr.sum()
        cumm_p_arr = np.cumsum(curr_p_arr)
        # randomly select additional n and recompute curr_size
        select_idx = np.unique(np.searchsorted(cumm_p_arr, np.random.random(add_n), side="right")).astype(np.int32)
        selected_mask[select_idx] = 1
        curr_size += selected_mask.sum()
    # return selected index
    return np.argwhere(selected_mask>0).T[0]

@jit(nopython=True)
def i_choose_j(i_coords, j_coords, n_arr, latlon, max_km=25, nlim=6, replace=True):
    """
    Select among the nearest n j-items for each i-item without replacement
    """
    # initialize
    i_idx_arr = np.arange(i_coords.shape[0], dtype=np.int32)
    j_idx_arr = np.arange(j_coords.shape[0], dtype=np.int32)
    # initialize result arr based on i (replace == True, then each i can choose any j)
    # or j (replace == False, then j can only be chosen once)
    if replace == True:
        result_arr = np.zeros(i_coords.shape[0], dtype=np.int32) - 1
    else:
        result_arr = np.zeros(j_coords.shape[0], dtype=np.int32) - 1

    for i in i_idx_arr:
        # n to sample from j
        n = n_arr[i]
        # get coordinates of i
        ix, iy = i_coords[i,:]
        # get j indices withint max_km of i
        min_ix = np.maximum(0, ix - max_km)
        max_ix = np.minimum(j_coords[:,0].max(), ix + max_km)
        min_iy = np.maximum(0, iy - max_km)
        max_iy = np.minimum(j_coords[:,1].max(), iy + max_km)
        j_shortlist = j_idx_arr[(j_coords[:,0]>=min_ix)&(j_coords[:,0]<=max_ix)&(j_coords[:,1]>=min_iy)&(j_coords[:,1]<=max_iy)]
        # sample without replacement - choose only those that has not been selected
        if replace == False:
            j_shortlist = j_shortlist[result_arr[j_shortlist]<0]
        # if we don't have enough, choose from nearest n + nlim
        if j_shortlist.size == n:
            j_selected = j_shortlist[:]
        else:
            if j_shortlist.size < n: # not enough n in shortlist
                # compute norms between i and all j
                j_norms = compute_norms(i_coords[i:i+1,], j_coords, latlon)[0]
                # select up to n+nlim as shortlist
                j_shortlist = j_idx_arr[np.argsort(j_norms)][:n+nlim]
                j_norms = j_norms[j_shortlist]
            else:
                # compute norms between i and j in shortlist
                j_norms = compute_norms(i_coords[i:i+1,], j_coords[j_shortlist,:], latlon)[0]

            # sort j_norms in ascending distance
            sorted_j_norms_idx = np.argsort(j_norms)
            j_shortlist = j_shortlist[sorted_j_norms_idx]
            j_norms = j_norms[sorted_j_norms_idx]
            # closer (smaller) distance = better score
            probs = 1 - (j_norms - j_norms.min())/(j_norms.max()-j_norms.min())
            # randomly select n number of j
            j_selected_idx = choose_nearest_idx_by_softmax(probs, n)
            j_selected = j_shortlist[j_selected_idx]

        # different result array depending on replacement
        if replace == False:
            for j in j_selected:
                result_arr[j] = i
        else:
            result_arr[i] = j_selected[0]

    # return of the corresponding i and j indexes that have been chosen
    if replace == False:
        j_selected = np.argwhere(result_arr>0).T[0].astype(np.int32)
        i_selected = result_arr[j_selected].astype(np.int32)
    else:
        i_selected = i_idx_arr[:].astype(np.int32)
        j_selected = result_arr[:].astype(np.int32)

    return i_selected, j_selected

@jit(nopython=True)
def get_inds_popgrid(location_arr, popgrid, inds):
    """
    Compute specific popgrid based on location of the given inds
    """
    loc_of_inds = location_arr[inds]
    sub_popgrid = np.zeros(popgrid.shape, dtype=np.int32)
    for k in np.arange(loc_of_inds.shape[0]):
        i,j = loc_of_inds[k]
        sub_popgrid[(i,j)] += 1
    return sub_popgrid

@jit(nopython=True)
def recompute_popgrid_within_commute_dist(popgrid, max_km):
    """
    Compute popgrid based on number of individuals within max_km
    """
    new_popgrid = np.zeros(popgrid.shape, dtype=np.int32)
    i_coords = np.arange(popgrid.shape[0])
    j_coords = np.arange(popgrid.shape[1])
    for i in i_coords:
        for j in j_coords:
            # get j indices withint max_km of i
            min_i = np.maximum(0, i - max_km)
            max_i = np.minimum(i_coords.max(), i + max_km)
            min_j = np.maximum(0, j - max_km)
            max_j = np.minimum(j_coords.max(), j + max_km)
            new_popgrid[i,j] = popgrid[min_i:max_i+1, min_j:max_j+1].sum()
    return new_popgrid

def get_edu_pars(datadir, country):
    # read education dataframes
    sch_enrol_rates_df = pd.read_csv(datadir+"UIS_school_enrollment_rates.csv")
    sch_entrance_age_df = pd.read_csv(datadir+"UIS_school_entrance_age.csv")
    sch_duration_df = pd.read_csv(datadir+"UIS_school_duration.csv")
    sch_teach_n_df = pd.read_csv(datadir+"UIS_school_teachers_n.csv")
    # get enrollment rate for country
    sch_enrol_rates_df = sch_enrol_rates_df[(sch_enrol_rates_df['LOCATION']==country)]
    sch_enrol_rates_df = sch_enrol_rates_df.dropna(subset=["Value"])
    sch_enrol_rates_df = sch_enrol_rates_df.set_index("Indicator")
    sch_enrol_rates = {}
    for idx in sch_enrol_rates_df.index.unique():
        row = sch_enrol_rates_df.loc[idx]
        if re.search("primary", idx, re.I):
            sch_enrol_rates[0] = np.float32(row[row['Time']==row['Time'].max()]['Value'].iloc[0]/100)
        elif re.search("lower secondary", idx, re.I):
            sch_enrol_rates[1] = np.float32(row[row['Time']==row['Time'].max()]['Value'].iloc[0]/100)
        elif re.search("upper secondary", idx, re.I):
            sch_enrol_rates[2] = np.float32(row[row['Time']==row['Time'].max()]['Value'].iloc[0]/100)
    # get enrollment age
    sch_entrance_age_df = sch_entrance_age_df[(sch_entrance_age_df['LOCATION']==country)]
    sch_entrance_age_df = sch_entrance_age_df.dropna(subset=["Value"])
    sch_entrance_age_df = sch_entrance_age_df.set_index("Indicator")
    sch_enrol_age = {}
    for idx in sch_entrance_age_df.index.unique():
        row = sch_entrance_age_df.loc[idx]
        if re.search("primary", idx, re.I):
            sch_enrol_age[0] = np.int32(row[row['Time']==row['Time'].max()]['Value'].iloc[0])
        elif re.search("lower secondary", idx, re.I):
            sch_enrol_age[1] = np.int32(row[row['Time']==row['Time'].max()]['Value'].iloc[0])
        elif re.search("upper secondary", idx, re.I):
            sch_enrol_age[2] = np.int32(row[row['Time']==row['Time'].max()]['Value'].iloc[0])
    # get final year age
    sch_duration_df = sch_duration_df[(sch_duration_df['LOCATION']==country)]
    sch_duration_df = sch_duration_df.dropna(subset=["Value"])
    sch_duration_df = sch_duration_df.set_index("Indicator")
    for idx in sch_duration_df.index.unique():
        row = sch_duration_df.loc[idx]
        if re.search("primary", idx, re.I):
            sch_enrol_age[0] = [sch_enrol_age[0], sch_enrol_age[0]+np.int32(row[row['Time']==row['Time'].max()]['Value'].iloc[0])-1]
        elif re.search("lower secondary", idx, re.I):
            sch_enrol_age[1] = [sch_enrol_age[1], sch_enrol_age[1]+np.int32(row[row['Time']==row['Time'].max()]['Value'].iloc[0])-1]
        elif re.search("upper secondary", idx, re.I):
            sch_enrol_age[2] = [sch_enrol_age[2], sch_enrol_age[2]+np.int32(row[row['Time']==row['Time'].max()]['Value'].iloc[0])-1]
    # get number of teachers
    sch_teach_n_df = sch_teach_n_df[(sch_teach_n_df['LOCATION']==country)]
    sch_teach_n_df = sch_teach_n_df.dropna(subset=["Value"])
    sch_teach_n_df = sch_teach_n_df.set_index("Indicator")
    sch_teach_n = {}
    for idx in sch_teach_n_df.index.unique():
        row = sch_teach_n_df.loc[idx]
        if re.search("primary", idx, re.I):
            sch_teach_n[0] = np.int32(row[row['Time']==row['Time'].max()]['Value'].iloc[0])
        elif re.search("lower secondary", idx, re.I):
            sch_teach_n[1] = np.int32(row[row['Time']==row['Time'].max()]['Value'].iloc[0])
        elif re.search("upper secondary", idx, re.I):
            sch_teach_n[2] = np.int32(row[row['Time']==row['Time'].max()]['Value'].iloc[0])
    return sch_enrol_rates, sch_enrol_age, sch_teach_n

def setup_entity_coords(entity_N, popgrid):
    """
    Distribution entities according to population density
    """
    # initialize entity_coords_arr
    entity_coords_arr = np.zeros((entity_N, 2), dtype=np.int32)
    # flatten popgrid
    flat_popgrid = popgrid.reshape(-1)
    # index flat_popgrid starting from one
    flat_popgrid_idx = np.arange(1, 1+flat_popgrid.size, dtype=np.int32)
    # get actual coordinates of popgrid
    popgrid_idx = flat_popgrid_idx.reshape(popgrid.shape)
    popgrid_idx = np.argwhere(popgrid_idx).astype(np.int32)

    # remove any places where there no people residing
    flat_popgrid_idx = flat_popgrid_idx[flat_popgrid>0]
    flat_popgrid = flat_popgrid[flat_popgrid>0]
    # compute pop density
    popden = flat_popgrid/flat_popgrid.sum()
    # select entity location by population density
    entity_location = np.random.choice(flat_popgrid_idx-1, size=entity_N, replace=True, p=popden)

    # randomly select location based on p_arr
    entity_i_coords = popgrid_idx[entity_location,0]
    entity_j_coords = popgrid_idx[entity_location,1]
    return entity_i_coords, entity_j_coords

def create_normal_rv_arr(min_size, mu, std, max_size, N):
    n = np.around(N/mu).astype(np.int32)
    rv_arr = np.int32(np.around(np.random.normal(loc=mu, scale=std, size=n), 0))
    rv_arr = rv_arr[(rv_arr>=min_size)&(rv_arr<=max_size)]
    arrsum = rv_arr.sum()
    diff = arrsum - N
    while diff != 0:
        if diff < 0: # total is higher than randomly selected sum
            n = np.int32(np.maximum(1, np.around(diff/mu)))
            add_rv_arr = np.int32(np.around(np.random.normal(loc=mu, scale=std, size=n), 0))
            add_rv_arr = add_rv_arr[(add_rv_arr>=min_size)&(add_rv_arr<=max_size)]
            rv_arr = np.concatenate((rv_arr, add_rv_arr))
        else: # total is lower than randomly selected sum
            for i in np.arange(len(rv_arr)):
                if rv_arr[i:].sum() <= N:
                    break
            rv_arr = rv_arr[i:]
            if rv_arr.sum() < N:
                add_rv_arr = np.asarray([N-rv_arr.sum()]).astype(np.int32)
                rv_arr = np.concatenate((rv_arr, add_rv_arr))
        arrsum = rv_arr.sum()
        diff = arrsum - N
    return rv_arr

def create_multinom_rv_arr(dist, bins, N):
    """
    Create random variable array based on values in bins which must total to N
    """
    mu = (dist * bins).sum() # expected entity size
    n = np.int32(np.around(N/mu))
    # calculate cummulative probability distribution
    cumm_p_arr = np.cumsum(dist)
    # randomly select n
    select_idx = np.searchsorted(cumm_p_arr, np.random.random(n), side="right").astype(np.int32)
    select_idx = select_idx[select_idx<dist.size]
    rv_arr = bins[select_idx].astype(np.int32)
    np.random.shuffle(rv_arr)
    diff = N - rv_arr.sum()
    while diff != 0:
        if diff > 0:
            # more to sample
            n = np.int32(np.maximum(1, np.around(diff/mu)))
            # randomly select n
            select_idx = np.searchsorted(cumm_p_arr, np.random.random(n), side="right").astype(np.int32)
            select_idx = select_idx[select_idx<dist.size]
            add_rv_arr = bins[select_idx].astype(np.int32)
            # add to rv_arr
            rv_arr = np.concatenate((rv_arr, add_rv_arr))
            np.random.shuffle(rv_arr)
        else:
            # remove from rv_arr
            for i in np.arange(len(rv_arr)):
                if rv_arr[i:].sum() <= N:
                    break
            rv_arr = rv_arr[i:]
            if rv_arr.sum() < N:
                add_rv_arr = np.asarray([N-rv_arr.sum()]).astype(np.int32)
                rv_arr = np.concatenate((rv_arr, add_rv_arr))
        diff = N - rv_arr.sum()
    return rv_arr

def get_hh_dist(datadir, country):
    """
    Get household size distribution 1-10 (currently curated by manually)
    """
    hh_dist_df = pd.read_csv(datadir+'world_hh_composition.csv')
    hh_dist_df = pd.read_csv("./data/world_hh_composition.csv").set_index("ISO3")
    hh_dist_df = hh_dist_df[list(map(str, np.arange(1, 11)))]
    try:
        hh_dist_arr = hh_dist_df.loc[country].to_numpy().astype(np.float32)
    except:
        raise Exception("%s not found in %s."%(country, datadir+'world_hh_composition.csv'))
    return hh_dist_arr

def get_age_dist(datadir, country):
    """
    Get age distribution of country based on UN World Population Prospects 2021
    """
    wpp_df = pd.read_excel(datadir+'WPP2022_POP_F02_1_POPULATION_5-YEAR_AGE_GROUPS_BOTH_SEXES.xlsx')
    # filter for country
    country_wpp_df = wpp_df[(wpp_df['ISO3 Alpha-code']==country)&(wpp_df['Year']==2021)]
    # extract age distribution in 5 year bins up to 99y
    agebins = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85-89', '90-94', '95-99', '100+']
    age_dist_arr = country_wpp_df[agebins].to_numpy()[0]
    age_dist_arr[19] += age_dist_arr[20]
    age_dist_arr = age_dist_arr[:20]
    age_dist_arr /= age_dist_arr.sum()
    return age_dist_arr.astype(np.float32)

def read_geotiff(fpath):
    """
    Read geotiff file and return population density grid
    """
    with rasterio.open(fpath) as src:
        # read data as numpy array (bands x rows x columns)
        data = src.read()[0]
        # get metadata
        metadata = src.meta
        # zero no data
        mask = data==metadata['nodata']
        data[mask] = 0.
        # remove any grid < 1
        data[data<1] = 0.
        # integer-ize
        data = np.around(data,0).astype(np.int32)

        # get latlon data
        latlon = np.zeros((data.shape[0], data.shape[-1], 2), dtype=np.float32)
        transform = src.transform
        i_coords = np.arange(data.shape[0])
        j_coords = np.arange(data.shape[-1])
        for i in i_coords:
            lon, lat = rasterio.transform.xy(transform, i, j_coords)
            latlon[i,:,0] = lat
            latlon[i,:,1] = lon
    return data, latlon

@jit(nopython=True)
def recalculate_popgrid(popgrid, location):
    new_popgrid = np.zeros(popgrid.shape, dtype=np.int32)
    for i in np.arange(location.shape[0]):
        x, y = location[i,:]
        new_popgrid[x,y] += 1
    return new_popgrid
