# import libraries
import re
import numpy as np
import scipy.sparse as sp
import sciris as sc
import pandas as pd

from numba import set_num_threads
from . import utils

# Surveillance object
class Surv():
    """
    Creates Surveillance Optimization object
    """
    def __init__(self, result_folder_path, healthsite_grid, pop, verbose=1):
        self.verbose = verbose
        # load popoobj
        self.pop = pop
        self.t = 0

        # translate to numpy array
        symptoms = sp.load_npz(result_folder_path + 'symptoms.npz').tocoo()
        self.ndays = symptoms.shape[-1]
        self.symp_inds = symptoms.row
        self.symp_days = symptoms.col
        self.symp_data = symptoms.data

        self.varhistory = sp.load_npz(result_folder_path + 'variant_history.npz').tocsc()
        self.nvar = np.unique(self.varhistory.data).size

        # flatten location
        self.flatten_location(healthsite_grid)

        # arrays
        self.tested_individuals_bool = np.zeros((self.nvar, self.pop.N), dtype=np.int8)

    def optimize_testing_loc(self):
        return

    def simulate_detectable_cases(self, mild_test_prob, sev_test_prob):
        """
        Simulate where cases are detectable 
        """
        detectable_cases = np.zeros((self.ndays, 2, self.nvar, self.n_locadd), dtype=np.int32)
        while self.t < self.ndays:
            # sample individuals that could be detectable today based on different testing probabilities
            detectable_cases_t = self.sample_inds(mild_test_prob, sev_test_prob)
            detectable_cases[self.t] = detectable_cases_t
            # update time-step
            self.t += 1
        return detectable_cases

    def sample_inds(self, mild_test_prob, sev_test_prob):
        detectable_cases_t = np.zeros((2, self.nvar, self.n_locadd), dtype=np.int32)
        # filter out symptomatic persons today
        symp_t_mask = self.symp_days == self.t
        symp_inds_t = self.symp_inds[symp_t_mask]
        if symp_inds_t.size == 0:
            return detectable_cases_t

        var_data_t = self.varhistory[:,self.t].tocoo().tocsr()[symp_inds_t].tocoo().data - 1
        # filter out those who have already been tested
        untested_bool = self.tested_individuals_bool[var_data_t, symp_inds_t] < 1

        symp_inds_t = symp_inds_t[untested_bool]
        var_data_t = var_data_t[untested_bool]
        symp_data_t = self.symp_data[symp_t_mask][untested_bool]

        for symptom in np.arange(2, dtype=np.int32):
            # separate individuals into mild or severe
            ms_symp_inds_t = symp_inds_t[symp_data_t == symptom+1]

            # testing probabilities
            if ms_symp_inds_t.size > 0:
                ms_var_data_t = var_data_t[symp_data_t == symptom+1]

                test_prob = sev_test_prob if symptom > 0 else mild_test_prob
                test_mask = np.random.random(ms_symp_inds_t.size) < test_prob

                ms_symp_inds_t = ms_symp_inds_t[test_mask]
                if ms_symp_inds_t.size > 0:
                    ms_var_data_t = ms_var_data_t[test_mask]

                    self.tested_individuals_bool[ms_var_data_t, ms_symp_inds_t] = 1
                    unique_locadd, locadd_idx, locadd_count = np.unique(self.pop.people.locadd[ms_symp_inds_t], return_index=True, return_counts=True)
                    detectable_cases_t[symptom, ms_var_data_t[locadd_idx], unique_locadd] += locadd_count

        return detectable_cases_t

    def flatten_location(self, healthsite_grid):
        """
        Flatten location to 1D address and pre-calculate distance between populated grids
        """
        if self.verbose > 0:
            print ('Flattening locations...')
        # flatten popgrid
        flat_popgrid = self.pop.popgrid.reshape(-1)
        # flatten healthsite
        flat_healthsite_grid = healthsite_grid.reshape(-1)

        # index flat_popgrid starting from one
        flat_popgrid_idx = np.arange(1, 1+flat_popgrid.size, dtype=np.int32)
        # convert individual location to flat index
        people_locidx = self.pop.people.location[:,0] * self.pop.popgrid.shape[-1] + self.pop.people.location[:,1] + 1
        # get actual coordinates of popgrid
        popgrid_idx = flat_popgrid_idx.reshape(self.pop.popgrid.shape)
        popgrid_idx = np.argwhere(popgrid_idx).astype(np.int32)

        # filter only for populated grids
        populated_flat_popgrid_idx = flat_popgrid_idx[(flat_popgrid > 0)|(flat_healthsite_grid > 0)]
        healthcare_flat_grid_idx = flat_popgrid_idx[flat_healthsite_grid > 0]

        populated_flat_popgrid_sortidx = np.arange(populated_flat_popgrid_idx.size, dtype=np.int32)
        # get people's location sortidx (renamed from here as location address)
        self.pop.people.locadd = populated_flat_popgrid_sortidx[np.searchsorted(populated_flat_popgrid_idx, people_locidx)]

        self.healthcare_locadd = populated_flat_popgrid_sortidx[np.searchsorted(populated_flat_popgrid_idx, healthcare_flat_grid_idx)]
        self.healthcare_n = flat_healthsite_grid[flat_healthsite_grid>0]

        # save location coordinates of populated grids and popgrid
        populated_grid_coords = popgrid_idx[populated_flat_popgrid_idx - 1]

        # compute haversine distance of each healthcare facilities to all populated grids
        self.populated_grid_to_hcf_dist = utils.compute_norms(populated_grid_coords, populated_grid_coords[self.healthcare_locadd], self.pop.latlon)
        self.n_locadd = self.populated_grid_to_hcf_dist.shape[0]
        return

# Simulation object
class Sim():
    """
    Creates Epidemic Simulation object
    """
    def __init__(self, pars):
        # load popoobj
        pars.popobj = sc.load(filename=pars.preload_datafolder + "pop.obj")
        # setup epidemic object
        self.epidemic = sc.objdict()
        # current state of each person
        self.epidemic.curr_state = np.zeros(pars.popobj.N, dtype=np.int8)
        # health states array (0: susceptible, 1: latent, 2: infectious, 3:dead, 4:non-infectious/recovered - wt, 5:non-infectious/recovered - mt)
        self.epidemic.state = sp.dok_matrix((pars.popobj.N, pars.ndays+1), dtype=np.int8)
        # symptom states array (0: asymptomatic, 1: symptomatic, 2:severe)
        self.epidemic.symptoms = sp.dok_matrix((pars.popobj.N, pars.ndays+1), dtype=np.int8)
        # transmission chain - save who infected whom
        nvar = 2 if pars.mt_bool > 0 else 1
        self.epidemic.transmission_history = sp.dok_matrix((pars.popobj.N, nvar), dtype=np.int32)
        # variant array
        self.epidemic.variant_history = sp.dok_matrix((pars.popobj.N, pars.ndays+1), dtype=np.int8)
        self.epidemic.curr_variant = np.zeros(pars.popobj.N, dtype=np.int8) - 1
        # save place of infection
        self.epidemic.infection_b_by_place = np.zeros(4, dtype=np.int32)
        # whether individuals will isolate if symptomatic
        self.epidemic.isolation = np.zeros(pars.popobj.N, dtype=np.int8)
        # population index array
        self.pop_arr = np.arange(pars.popobj.N, dtype=np.int32)
        self.people = pars.popobj.people
        # get average contact mat
        self.household_contact_mat = utils.get_others_contact_mat(pars.datadir, pars.country, 'home')
        self.school_contact_mat = utils.get_others_contact_mat(pars.datadir, pars.country, 'school')
        self.work_contact_mat = utils.get_others_contact_mat(pars.datadir, pars.country, 'work')
        self.other_contact_mat = utils.get_others_contact_mat(pars.datadir, pars.country, 'others')

        ## -- prepare run matrix and arrays -- ##
        # flatten location to 1D address (for easy reference) and filter to retain only populated grids
        pars = self.flatten_location(pars)
        # extract parameters for pathogen
        pars = self.extract_pars(pars)

        # initialise epidemic
        self.day = np.int32(0) # current day
        self.var_inf_counts = np.zeros(2, dtype=np.int32)
        self.date = pd.to_datetime(pars.startdate)
        self.initialise(pars.init_popden_p, pars.init_n, 0, pars)

        # set num_threads
        #set_num_threads(pars.num_threads)
        # simulate
        self.simulate(pars)

    def transmission(self, pars):

        # get all infectious individuals
        infectious_inds = self.pop_arr[(self.epidemic.curr_state == 2)]
        if len(infectious_inds) == 0:
            # return if no infectious individuals
            return
        infectious_vars = self.epidemic.curr_variant[infectious_inds]
        infectious_agebin = self.people.agebin[infectious_inds]

        # get symptoms of infectious persons
        symp_of_infectious_inds = self.epidemic.symptoms[infectious_inds, self.day].toarray().T[0]
        iso_red_f = np.ones(len(symp_of_infectious_inds), dtype=np.float32)
        # isolate those that would
        iso_red_f[(self.epidemic.isolation[infectious_inds]>0)&(symp_of_infectious_inds == 1)] -= pars.isolate_mild_trans_red[0]
        iso_red_f[(self.epidemic.isolation[infectious_inds]>0)&(symp_of_infectious_inds == 2)] -= pars.isolate_sev_trans_red

        # get mask of susceptible people
        if pars.mt_bool > 0:
            sus_mask = ((self.epidemic.curr_state == 0)|(self.epidemic.curr_state == 4))
        else:
            sus_mask = self.epidemic.curr_state == 0

        ## -- household infections -- ##
        # get overdispersion factor of infectious individuals
        if pars.overdispersion_bool > 0:
            overdispersion_f = utils.negbin(mean=pars.overdispersion_pars[0], shape=pars.overdispersion_pars[1], size=len(infectious_inds))
        else:
            overdispersion_f = np.ones(infectious_inds.size, dtype=np.float32)
        # get infectious households
        infectious_places = self.people.household[infectious_inds]
        # get susceptible individuals in these households
        susceptible_inds = self.pop_arr[np.isin(self.people.household, infectious_places)&sus_mask]
        susceptible_states = self.epidemic.curr_state[susceptible_inds]
        susceptible_inds_agebin = self.people.agebin[susceptible_inds]
        susceptible_places = self.people.household[susceptible_inds]
        # get relative susceptiblity by age of susceptible individuals
        relsus_f = utils.rand_generator(pars.rel_sus, ind_agebin=susceptible_inds_agebin, vtype=np.float32)
        # get protection of susceptibles from previous immunity
        protection_from_prev_immunity_p = 1 - utils.rand_generator(pars.immunity, ind_agebin=susceptible_inds_agebin, vtype=np.float32)
        # compute transmission
        contact_mat = self.household_contact_mat
        sus_infected_mask, sus_infector = utils.compute_transmission(infectious_inds, infectious_places, overdispersion_f, iso_red_f, susceptible_places, relsus_f, protection_from_prev_immunity_p, pars.beta, pars.setting_beta_f[0], contact_mat, infectious_agebin, susceptible_inds_agebin, infectious_vars, susceptible_states, pars.mt_trans_f, pars.cross_wt_immunity)
        # filter for infected persons
        exposed_persons = susceptible_inds[sus_infected_mask>0]

        if len(exposed_persons) > 0:
            # get transmitted variant
            trans_variants = sus_infected_mask[sus_infected_mask>0] - 1
            # save who infected exposed person
            infectors = sus_infector[sus_infected_mask>0]
            self.epidemic.transmission_history[exposed_persons, trans_variants] = infectors
            self.epidemic.infection_b_by_place[0] += exposed_persons.size
            # plan infection
            self.exposed(exposed_persons, trans_variants, pars)

        ## -- school/workplace infections -- ##
        # get overdispersion factor of infectious individuals
        if pars.overdispersion_bool > 0:
            overdispersion_f = utils.negbin(mean=pars.overdispersion_pars[0], shape=pars.overdispersion_pars[1], size=len(infectious_inds))
        else:
            overdispersion_f = np.ones(infectious_inds.size, dtype=np.float32)
        if self.date.weekday() < 5: # weekdays
            for place_type in [1, 2]:
                # 1 = school, 2 = workplaces
                # get infectious places
                infectious_places = self.people.place_id[infectious_inds]
                place_type_mask = self.people.place[infectious_inds] == place_type # remove any infectious individuals with no associated place
                # get susceptible individuals in these places
                susceptible_inds = self.pop_arr[(np.isin(self.people.place_id, infectious_places[place_type_mask]))&sus_mask]
                susceptible_states = self.epidemic.curr_state[susceptible_inds]
                susceptible_inds_agebin = self.people.agebin[susceptible_inds]
                susceptible_places = self.people.place_id[susceptible_inds]
                # get relative susceptiblity by age of susceptible individuals
                relsus_f = utils.rand_generator(pars.rel_sus, ind_agebin=susceptible_inds_agebin, vtype=np.float32)
                # get protection of susceptibles
                protection_from_prev_immunity_p = 1 - utils.rand_generator(pars.immunity, ind_agebin=susceptible_inds_agebin, vtype=np.float32)
                # get contact matrix
                if place_type > 1:
                    contact_mat = self.work_contact_mat
                    iso_red_f = np.ones(len(symp_of_infectious_inds), dtype=np.float32)
                    iso_red_f[(self.epidemic.isolation[infectious_inds]>0)&(symp_of_infectious_inds == 1)] -= pars.isolate_mild_trans_red[place_type]
                    iso_red_f[(self.epidemic.isolation[infectious_inds]>0)&(symp_of_infectious_inds == 2)] -= pars.isolate_sev_trans_red
                else:
                    contact_mat = self.school_contact_mat
                    # assume that students would not go to school if symptomatic
                    iso_red_f = np.ones(len(symp_of_infectious_inds), dtype=np.float32)
                    iso_red_f[(self.epidemic.isolation[infectious_inds]>0)&(symp_of_infectious_inds == 1)] -= pars.isolate_mild_trans_red[place_type]
                    iso_red_f[(self.epidemic.isolation[infectious_inds]>0)&(symp_of_infectious_inds == 2)] -= pars.isolate_sev_trans_red

                # compute transmission
                sus_infected_mask, sus_infector = utils.compute_transmission(infectious_inds[place_type_mask], infectious_places[place_type_mask], overdispersion_f[place_type_mask], iso_red_f[place_type_mask], susceptible_places, relsus_f, protection_from_prev_immunity_p, pars.beta, pars.setting_beta_f[place_type], contact_mat, infectious_agebin[place_type_mask], susceptible_inds_agebin, infectious_vars[place_type_mask], susceptible_states, pars.mt_trans_f, pars.cross_wt_immunity)

                # filter for infected persons
                exposed_persons = susceptible_inds[sus_infected_mask>0]

                if len(exposed_persons) > 0:
                    # get transmitted variant
                    trans_variants = sus_infected_mask[sus_infected_mask>0] - 1
                    # save who infected exposed person
                    infectors = sus_infector[sus_infected_mask>0]
                    self.epidemic.transmission_history[exposed_persons, trans_variants] = infectors
                    self.epidemic.infection_b_by_place[place_type] += exposed_persons.size
                    # plan infection
                    self.exposed(exposed_persons, trans_variants, pars)

        ## -- random transmissions -- ##
        # get overdispersion factor of infectious individuals
        if pars.overdispersion_bool > 0:
            overdispersion_f = utils.negbin(mean=pars.overdispersion_pars[0], shape=pars.overdispersion_pars[1], size=len(infectious_inds))
        else:
            overdispersion_f = np.ones(infectious_inds.size, dtype=np.float32)
        # identify which susceptibles could interact with infectious individuals randomly
        # get infectious locadd
        infectious_locadd = self.people.locadd[infectious_inds]
        # reduction
        iso_red_f = np.ones(len(symp_of_infectious_inds), dtype=np.float32)
        iso_red_f[(self.epidemic.isolation[infectious_inds]>0)&(symp_of_infectious_inds == 1)] -= pars.isolate_mild_trans_red[3]
        iso_red_f[(self.epidemic.isolation[infectious_inds]>0)&(symp_of_infectious_inds == 2)] -= pars.isolate_sev_trans_red
        # get susceptible individuals in these places
        susceptible_inds = self.pop_arr[sus_mask]
        susceptible_locadd = self.people.locadd[susceptible_inds]
        susceptible_inds = utils.identify_rand_sus(infectious_locadd, susceptible_locadd, susceptible_inds, self.commute_kernel, self.ind_commute_kernel_sum)

        if susceptible_inds.size > 0:
            susceptible_locadd = self.people.locadd[susceptible_inds]
            susceptible_inds_agebin = self.people.agebin[susceptible_inds]
            susceptible_states = self.epidemic.curr_state[susceptible_inds]
            # get relative susceptiblity by age of susceptible individuals
            relsus_f = utils.rand_generator(pars.rel_sus, ind_agebin=susceptible_inds_agebin, vtype=np.float32)
            # get protection of susceptibles
            protection_from_prev_immunity_p = 1 - utils.rand_generator(pars.immunity, ind_agebin=susceptible_inds_agebin, vtype=np.float32)
            # get contact matrix
            contact_mat = self.other_contact_mat
            # compute transmission
            sus_infected_mask, sus_infector = utils.compute_rand_transmission(infectious_inds, infectious_locadd, overdispersion_f, iso_red_f, susceptible_locadd, relsus_f, protection_from_prev_immunity_p, pars.beta, pars.setting_beta_f[3], contact_mat, self.commute_kernel, self.ind_commute_kernel_sum, infectious_agebin, susceptible_inds_agebin, susceptible_inds, infectious_vars, susceptible_states, pars.mt_trans_f, pars.cross_wt_immunity)
            # filter for infected persons
            exposed_persons = susceptible_inds[sus_infected_mask>0]
            if len(exposed_persons) > 0:
                # get transmitted variant
                trans_variants = sus_infected_mask[sus_infected_mask>0] - 1
                # save who infected exposed person
                infectors = sus_infector[sus_infected_mask>0]
                self.epidemic.transmission_history[exposed_persons, trans_variants] = infectors
                self.epidemic.infection_b_by_place[3] += exposed_persons.size
                # plan infection
                self.exposed(exposed_persons, trans_variants, pars)
        return

    def simulate(self, pars):
        while self.day < pars.ndays+1:
            if pars.mt_bool > 0 and self.day == pars.mt_intro:
                # introduce mutant
                self.initialise(pars.mt_init_popden_p, pars.mt_init_n, 1, pars)
            # update current state
            self.update_curr_state(pars.verbose)
            # transmission
            self.transmission(pars)
            # update time-step
            self.day += 1
            self.date += pd.Timedelta(days=1)
        return

    def initialise(self, init_popden_p, init_n, vartype, pars):
        """
        Initialise simulation object
        """
        # randomly select where initial infection(s) occur from potential location address
        lowmit = init_popden_p
        uppmit = np.minimum(lowmit + 0.1, 1.)
        m1 = self.populated_sus_locadd.sum(axis=1) > np.quantile(self.populated_sus_locadd.sum(axis=1), lowmit)
        m2 = self.populated_sus_locadd.sum(axis=1) <= np.quantile(self.populated_sus_locadd.sum(axis=1), uppmit)
        init_inf_locadd = np.random.choice(self.populated_grid_locadd[m1&m2])
        self.init_inf_loc = self.populated_grid_coords[init_inf_locadd] # save where initial infection location is
        # randomly select initially infected persons from individuals in location
        candidate_persons = self.pop_arr[(self.people.locadd == init_inf_locadd)&(self.epidemic.curr_state==0)]
        # initialize infection
        init_n = np.minimum(candidate_persons.size, init_n)
        exposed_persons = np.random.choice(candidate_persons, init_n, replace=False)
        trans_variants = np.zeros(exposed_persons.size, dtype=np.int8) + vartype
        # plan infection of initially infected person
        self.exposed(exposed_persons, trans_variants, pars)

    def exposed(self, exposed_persons, trans_variants, pars):

        n = exposed_persons.size
        exposed_agebin = self.people.agebin[exposed_persons]
        # update populated_sus_locadd
        exposed_locadd = self.people.locadd[exposed_persons]

        ## -- plan symptom trajectory -- ##
        symp_presentation = np.ones(n, dtype=np.int8)
        # randomly select those who will be asymptomatic
        # get asymptomatic probability by age
        asymp_prob = utils.rand_generator(pars.asymp_prob, ind_agebin=exposed_agebin, vtype=np.float32)
        symp_presentation[np.random.random(n) < asymp_prob] = 0
        symp_prob = 1. - asymp_prob
        # randomly select those with severe symptoms
        n_symp = symp_presentation.sum()
        sev_presentation = np.zeros(n, dtype=np.int8)
        dead_mask = np.zeros(n, dtype=np.int8)

        if n_symp > 0:
            # determine who will have severe symptoms
            sev_prob = utils.rand_generator(pars.sev_prob, ind_agebin=exposed_agebin, vtype=np.float32)
            sev_presentation[(symp_presentation>0)&(np.random.random(n) < sev_prob/symp_prob)] = 1
            n_sev = sev_presentation.sum()
            # whether these mild symptomatic individuals would isolate
            isolate_symp_mask = np.random.random(n_symp - n_sev) < pars.redtranslike
            # isolate mild individuals who will and severe individuals
            self.epidemic.isolation[exposed_persons[(symp_presentation>0)&(sev_presentation==0)][isolate_symp_mask]] = 1
            self.epidemic.isolation[exposed_persons[sev_presentation>0]] = 1

            # randomly select those that will die
            if n_sev > 0:
                dea_prob = utils.rand_generator(pars.dea_prob, ind_agebin=exposed_agebin, vtype=np.float32)
                dead_mask[(sev_presentation>0)&(np.random.random(n) < dea_prob/sev_prob)] = 1

        ## -- plan infection trajectory -- ##
        # get period of proliferation
        proliferation_tau = self.day + utils.rand_generator(pars.proliferation_tau, ind_agebin=exposed_agebin, minval=1, vtype=np.int32)
        # get period of clearance
        clearance_tau = proliferation_tau + utils.rand_generator(pars.clearance_tau, ind_agebin=exposed_agebin, minval=1, vtype=np.int32)
        # compute latent and infectious phase
        latent_tau, infectious_tau = utils.compute_latent_infectious_trajectories(start_t=self.day, peak_t_arr=proliferation_tau, end_t_arr=clearance_tau, start_vload=pars.start_vload['par1'][exposed_agebin], peak_vload=pars.peak_vload['par1'][exposed_agebin], inf_vload=pars.inf_vload['par1'][exposed_agebin])

        max_day = np.minimum(clearance_tau.max()+1, pars.ndays+1)
        for d in np.arange(self.day, max_day):
            # state - latent
            mask = d<latent_tau
            self.epidemic.state[exposed_persons[mask], d] = 1
            self.epidemic.variant_history[exposed_persons[mask], d] = trans_variants[mask] + 1

            # state - infectious
            mask = (d>=latent_tau)&(d<infectious_tau)
            self.epidemic.state[exposed_persons[mask], d] = 2
            self.epidemic.variant_history[exposed_persons[mask], d] = trans_variants[mask] + 1

            # state - non-infectious/recovered
            mask = (d>=infectious_tau)&(d<clearance_tau)
            state_inds = exposed_persons[mask]
            nir_state = np.zeros(state_inds.size, dtype=np.int8) + 4
            nir_state[trans_variants[mask] > 0] += 1
            self.epidemic.state[state_inds, d] = nir_state
            self.epidemic.variant_history[state_inds, d] = trans_variants[mask] + 1

            # # state - end of clearance
            # death
            mask =(d==clearance_tau)&(dead_mask>0)
            self.epidemic.state[exposed_persons[mask], d] = 3 # dead
            self.epidemic.variant_history[exposed_persons[mask], d] = trans_variants[mask] + 1

            # remain recovered
            mask = (d==clearance_tau)&(dead_mask<1)
            state_inds = exposed_persons[mask]
            nir_state = np.zeros(state_inds.size, dtype=np.int8) + 4
            nir_state[trans_variants[mask] > 0] += 1
            self.epidemic.state[state_inds, d] = nir_state
            self.epidemic.variant_history[state_inds, d] = trans_variants[mask] + 1

            # (degree of) symptoms (if any) starts on end of proliferation onwards
            self.epidemic.symptoms[exposed_persons[(symp_presentation>0)&(d>=proliferation_tau)&(d<clearance_tau)], d] = 1
            self.epidemic.symptoms[exposed_persons[(sev_presentation>0)&(d>=proliferation_tau)&(d<clearance_tau)], d] = 2

        # update curr_state of exposed_persons today
        self.epidemic.curr_state[exposed_persons] = 1
        # remember what variant they were infected by
        self.epidemic.curr_variant[exposed_persons] = trans_variants
        self.var_inf_counts[0] += trans_variants[trans_variants == 0].size
        self.var_inf_counts[1] += trans_variants[trans_variants == 1].size
        #self.epidemic.variant_history[exposed_persons, self.day] = trans_variants + 1

    def update_curr_state(self, verbose):
        # update curr_state
        persons_with_status = self.epidemic.state[:,self.day].tocoo().row
        status_of_persons = self.epidemic.state[:,self.day].tocoo().data
        self.epidemic.curr_state[persons_with_status] = status_of_persons
        # print details
        if verbose > 0:
            # differentiate by status
            status_arr = np.zeros(6, dtype=np.int32)
            status, counts = np.unique(self.epidemic.curr_state, return_counts=True)
            status_arr[status] = counts
            # differentiate by variant
            print (self.day, self.date.strftime('%Y-%m-%d'), status_arr, self.var_inf_counts, self.epidemic.infection_b_by_place, self.populated_sus_locadd.sum())

    def extract_pars(self, pars):
        df = pd.read_excel(pars.datadir+"parameters.xlsx")
        # filter for pathogen
        df = df[df['pathogen']==pars.pathogen].set_index(['par', 'min_age', 'max_age'])
        for (parameter, min_age, max_age) in df.index.unique():
            # get dist
            pardist = df.loc[(parameter, min_age, max_age), 'dist']
            # extract age range of estimate
            min_agebin = np.floor(np.int32(min_age)/5).astype(np.int32)
            max_agebin = np.floor(np.int32(max_age)/5).astype(np.int32)
            # grab type of dist and par
            try:
                pars[parameter]['dist'] = df.loc[(parameter, min_age, max_age), 'dist']
            except:
                pars[parameter] = {}
                pars[parameter]['dist'] = df.loc[(parameter, min_age, max_age), 'dist']

            try:
                pars[parameter]['par1'][min_agebin:max_agebin+1] = df.loc[(parameter, min_age, max_age), 'par1']
            except:
                pars[parameter]['par1'] = np.zeros(20, dtype=np.float32) - 1
                pars[parameter]['par1'][min_agebin:max_agebin+1] = df.loc[(parameter, min_age, max_age), 'par1']

            # par2 only for those appropriate dist
            if pars[parameter]['dist'] in ['uniform', 'lognormal', 'normal', 'negbin']:
                try:
                    pars[parameter]['par2'][min_agebin:max_agebin+1] = df.loc[(parameter, min_age, max_age), 'par2']
                except:
                    pars[parameter]['par2'] = np.zeros(20, dtype=np.float32) - 1
                    pars[parameter]['par2'][min_agebin:max_agebin+1] = df.loc[(parameter, min_age, max_age), 'par2']

        return pars

    def flatten_location(self, pars):
        """
        Flatten location to 1D address and pre-calculate distance between populated grids
        """
        if pars.verbose > 0:
            print ('Flattening locations...')
        # flatten popgrid
        flat_popgrid = pars.popobj.popgrid.reshape(-1)
        # index flat_popgrid starting from one
        flat_popgrid_idx = np.arange(1, 1+flat_popgrid.size, dtype=np.int32)
        # convert individual location to flat index
        people_locidx = self.people.location[:,0] * pars.popobj.popgrid.shape[-1] + self.people.location[:,1] + 1
        # get actual coordinates of popgrid
        popgrid_idx = flat_popgrid_idx.reshape(pars.popobj.popgrid.shape)
        popgrid_idx = np.argwhere(popgrid_idx).astype(np.int32)
        # filter only for populated grids
        populated_flat_popgrid_idx = flat_popgrid_idx[flat_popgrid>0]
        populated_flat_popgrid_sortidx = np.arange(populated_flat_popgrid_idx.size, dtype=np.int32)
        # get people's location sortidx (renamed from here as location address)
        self.people.locadd = populated_flat_popgrid_sortidx[np.searchsorted(populated_flat_popgrid_idx, people_locidx)]
        # save location coordinates of populated grids and popgrid
        self.populated_grid_coords = popgrid_idx[populated_flat_popgrid_idx-1]
        self.populated_grid_locadd = populated_flat_popgrid_sortidx
        # get distribution of susceptible people across populated grids
        self.populated_sus_locadd = utils.get_age_dist_of_grids(self.populated_grid_locadd, self.people.locadd, self.people.agebin)
        self.populated_flat_popgrid = self.populated_sus_locadd.sum(axis=1).astype(np.int32)

        self.commute_kernel = np.zeros((self.populated_grid_locadd.size, self.populated_grid_locadd.size), dtype=np.float32)
        # get gravity probability distribution between populated grids based on distance
        if pars.verbose > 0:
            print ('Pre-calculating commuting probability between locations...')
        # compute distance kernel based on haversine distance
        commute_par1, commute_par2 = pars.commute_pars
        self.commute_kernel, self.ind_commute_kernel_sum = utils.compute_commute_kernel(self.populated_grid_coords, pars.popobj.latlon, self.people.locadd, commute_par1, commute_par2)

        return pars

# Population object
class Pop():
    """
    Creates population object
    """
    def __init__(self, pars=None):
        # set num_threads
        #set_num_threads(pars.num_threads)
        # get population grid and lonlat
        self.popgrid, self.latlon = utils.read_geotiff(pars.datadir+"worldpop/%s_ppp_2020_1km_Aggregated_UNadj.tif"%(pars.country))
        # total population size
        self.N = self.popgrid.sum()
        # initialise people
        self.people = sc.objdict()
        # setup age of individuals
        self.setup_age(pars)
        # setup residence in popgrid
        self.setup_residence(pars)
        # setup schools
        self.setup_schools(pars)
        # setup workplaces
        self.setup_workplaces(pars)

    def setup_workplaces(self, pars):
        """
        Setup workplaces
        """
        if pars.verbose > 0:
            print ('Setting up workplaces...')
        # get employment force
        pop_arr = np.arange(self.people.age.size, dtype=np.int32)
        labour_force = pop_arr[(self.people.age>=pars.min_employment_age)&(self.people.age<=pars.retirement_age)]
        # calculate expected number of employed individuals
        n_employed = np.around(labour_force.size * pars.employment_rate).astype(np.int32)
        # check that we have enough employed individuals
        if n_employed < pars.workplaces_n:
            raise Exception("Too many workplaces and not enough employees.")
        # remove those that belong to school already
        lab_mask = self.people.place_id[labour_force]>=0
        labour_force = labour_force[~lab_mask]
        # randomly select individuals at work
        employed_inds = np.random.choice(labour_force, n_employed, replace=False)
        employed_loc = self.people.location[employed_inds]
        # randomly pick individual locations as workplace locations
        employed_inds_idx = np.arange(employed_inds.size, dtype=np.int32)
        selected_employed_inds_idx = np.random.choice(employed_inds_idx, pars.workplaces_n, replace=False)
        selected_employed_inds = employed_inds[selected_employed_inds_idx]
        # get their locations
        workplace_locs = self.people.location[selected_employed_inds]
        workplace_ids = np.arange(workplace_locs.shape[0], dtype=np.int32)
        workplace_size = np.ones(workplace_ids.size, dtype=np.float32)
        # assign sellected employed inds their workplace
        employed_inds_placement = np.zeros(employed_inds.size, dtype=np.int32) - 1
        employed_inds_placement[selected_employed_inds_idx] = workplace_ids
        # count n_unassigned
        n_unassigned = employed_inds_placement[employed_inds_placement<0].size

        same_bool = 0
        for step in np.arange(1000):
            # get randomized growth rate this step
            growth_rate_arr = np.random.lognormal(mean=pars.wp_growth_lognorm_pars[0], sigma=pars.wp_growth_lognorm_pars[1], size=workplace_ids.size).astype(np.float32)
            # grow workplace
            # take the maximum of current or growing workplace sizes
            step_workplace_size = np.maximum(workplace_size, workplace_size*growth_rate_arr)
            # round the potential growth
            potential_workplace_growth_n = np.around(step_workplace_size - workplace_size,0).astype(np.int32)
            # get all workplaces that grow this step
            growing_workplace_id = workplace_ids[potential_workplace_growth_n > 0]
            # update workplace sizes that did not grow
            workplace_size[workplace_ids[potential_workplace_growth_n<1]] = step_workplace_size[workplace_ids[potential_workplace_growth_n<1]]
            # get the growth per workplace this step
            potential_workplace_growth_n = potential_workplace_growth_n[potential_workplace_growth_n>0]
            # verbose
            if pars.verbose > 1:
                print ('Growing workplace, step %i: '%(step+1), 'Unassigned = %i; '%(n_unassigned), "Add = %i"%(potential_workplace_growth_n.sum()))
            # randomly assign individuals to workplace
            # get all employees who have not been placed a workplace
            curr_employed_inds_idx = employed_inds_idx[employed_inds_placement<0]
            # sample these employees
            samp_n = potential_workplace_growth_n.sum()
            if samp_n >= curr_employed_inds_idx.size:
                curr_selected_employed_inds_idx = curr_employed_inds_idx[:]
            else:
                curr_selected_employed_inds_idx = np.random.choice(curr_employed_inds_idx, samp_n, replace=False)
            # assign them workplaces
            curr_assigned_workplace_id = utils.workplace_choose_ind(growing_workplace_id, workplace_locs[growing_workplace_id], employed_loc[curr_selected_employed_inds_idx], potential_workplace_growth_n, pars.max_work_commute_dist)
            employed_inds_placement[curr_selected_employed_inds_idx] = curr_assigned_workplace_id[:]
            # update workplace size
            workplace_size = utils.update_arr_counts(workplace_size, curr_assigned_workplace_id)
            # count n_unassigned
            new_n_unassigned = employed_inds_placement[employed_inds_placement<0].size
            # break conditions
            if new_n_unassigned == 0:
                break
            if new_n_unassigned < n_unassigned:
                n_unassigned = new_n_unassigned
                same_bool = 0
            else:
                same_bool += 1
                if (same_bool > 5) and new_n_unassigned/employed_inds.size < 0.03:
                    break
        if pars.verbose > 0:
            print ('...done.')
        # assign individuals
        mask = employed_inds_placement >= 0
        employed_inds = employed_inds[mask]
        self.people.place[employed_inds] = 2
        self.people.place_id[employed_inds] = employed_inds_placement[mask]
        self.people.place_loc[employed_inds] = workplace_locs[employed_inds_placement[mask]]
        return #employed_inds_placement[mask]

    def setup_schools(self, pars):
        """
        Setup schools
        """
        if pars.verbose > 0:
            print ('Setting up schools...')
        # get school information arrays
        sch_enrol_rates, sch_enrol_age, sch_teach_n = utils.get_edu_pars(pars.datadir, pars.country.upper())
        pop_arr = np.arange(self.people.age.size, dtype=np.int32)
        # initialise place for people (1: school, 2: workplace)
        self.people.place = np.zeros(pop_arr.size, dtype=np.int8)
        # initialise place id and location for people
        self.people.place_id = np.zeros(pop_arr.size, dtype=np.int32) - 1
        self.people.place_loc = np.zeros((pop_arr.size, 2), dtype=np.int32) - 1
        # prime adults with potential to be teachers
        prime_adults = pop_arr[(self.people.age>=pars.prime_age)&(self.people.age<=pars.retirement_age)]
        # get students and teachers at each level
        max_place_id = np.int32(0)

        for slvl in np.arange(2):
            # get number of individuals attending this level of education
            min_age, max_age = sch_enrol_age[slvl]
            mask = (self.people.age>=min_age)&(self.people.age<=max_age)
            # number of students
            slvl_n = np.int32(np.around(sch_enrol_rates[slvl] * mask.sum(), 0))
            # randomly select students
            slvl_inds = np.random.choice(pop_arr[mask], slvl_n, replace=False)
            # number of teachers
            slvl_teach_n = np.int32(sch_teach_n[slvl])
            # secondary = lower and upper
            if slvl > 0:
                min_age, max_age = sch_enrol_age[slvl+1]
                mask = (self.people.age>=min_age)&(self.people.age<=max_age)
                add_slvl_n = np.int32(np.around(sch_enrol_rates[slvl+1] * mask.sum(), 0))
                add_slvl_inds = np.random.choice(pop_arr[mask], add_slvl_n, replace=False)
                slvl_n += add_slvl_n
                slvl_inds = np.concatenate((slvl_inds, add_slvl_inds))
                slvl_teach_n += np.int32(sch_teach_n[slvl+1])
            # get students per teacher ratio
            slvl_teach_per_stu = slvl_teach_n/slvl_n
            # get array of school sizes (currently only meant to determine the number of schools there are)
            min_size, mu, std, max_size = pars.school_size_pars
            sch_size_arr = utils.create_normal_rv_arr(min_size, mu, std, max_size, slvl_n)
            # compute sub-popgrid based on potential students of this level
            sub_popgrid = utils.get_inds_popgrid(self.people.location, self.popgrid, slvl_inds)
            """
            # re-compute sub_popgrid based on max distance travelled by students
            sub_popgrid = utils.recompute_popgrid_within_commute_dist(sub_popgrid, pars.max_sch_commute_dist)
            """
            # get location coordinates of school, distributing them by population density
            sch_i_coords, sch_j_coords = utils.setup_entity_coords(sch_size_arr.size, sub_popgrid)
            # let each student choose which school to go to by gravity model
            sch_coords = np.asarray([sch_i_coords, sch_j_coords]).T
            ind_coords = self.people.location[slvl_inds]
            chosen_sch_idx = utils.i_choose_j(ind_coords, sch_coords, np.ones(ind_coords.size, dtype=np.int32), self.latlon, max_km=pars.max_sch_commute_dist, replace=True)[1]
            # assign students to school
            self.people.place[slvl_inds] = 1
            self.people.place_id[slvl_inds] = max_place_id + chosen_sch_idx
            self.people.place_loc[slvl_inds] = sch_coords[chosen_sch_idx,:]
            # get unique school coordinates and number of students in each of them
            unique_chosen_sch, unique_chosen_sch_idx, unique_chosen_sch_stu_size = np.unique(chosen_sch_idx, return_index=True, return_counts=True)
            unique_chosen_sch = max_place_id + unique_chosen_sch
            unique_sch_coords = sch_coords[chosen_sch_idx[unique_chosen_sch_idx]]
            max_place_id = np.unique(unique_chosen_sch).max()+1
            # compute the number of teachers in each school (poisson)
            teachers_n_in_each_unique_sch = np.around(unique_chosen_sch_stu_size * slvl_teach_per_stu, 0).astype(np.int32)
            teachers_n_in_each_unique_sch = np.maximum(1, teachers_n_in_each_unique_sch)
            # select adults within the vicinity as teachers
            unemployed_prime_adults = prime_adults[self.people.place[prime_adults]<1]
            # let each school choose a number of teachers by gravity model
            unemployed_prime_adults_loc = self.people.location[unemployed_prime_adults]
            chosen_sch_idx, chosen_teachers_idx = utils.i_choose_j(unique_sch_coords, unemployed_prime_adults_loc, teachers_n_in_each_unique_sch, self.latlon, max_km=pars.max_work_commute_dist, replace=False)
            # assign chosen teachers to school
            chosen_teachers = unemployed_prime_adults[chosen_teachers_idx]
            self.people.place[chosen_teachers] = 1
            self.people.place_id[chosen_teachers] = unique_chosen_sch[chosen_sch_idx]
            self.people.place_loc[chosen_teachers] = unique_sch_coords[chosen_sch_idx,:]
            # verbose
            if pars.verbose > 1:
                print ('No. of students (%s): %i'%('primary' if slvl == 0 else 'secondary', slvl_inds.size))
                print ('No. of schools (%s): %i'%('primary' if slvl == 0 else 'secondary', unique_chosen_sch.size))
                print ('No. of teachers (%s): %i'%('primary' if slvl == 0 else 'secondary', chosen_teachers.size))

        if pars.verbose > 0:
            sch_inds = pop_arr[self.people.place == 1]
            teachers_inds = sch_inds[(self.people.age[sch_inds]>=pars.prime_age)&(self.people.age[sch_inds]<=pars.retirement_age)]
            students_inds = sch_inds[(self.people.age[sch_inds]<=max_age)]
            place_id_arr = np.unique(self.people.place_id[self.people.place_id>=0])
            print ('Total school population (pri. and sec.): %i'%(sch_inds.size))
            print ('Total no. of students (pri. and sec.): %i'%(students_inds.size))
            print ('Total no. of schools (pri. and sec.): %i'%(place_id_arr.size))
            print ('Total no. of teachers (pri. and sec.): %i'%(teachers_inds.size))
            print ('...done.')

    def setup_residence(self, pars):
        """
        Setup households and residence in grid
        """
        if pars.verbose > 0:
            print ('Setting up residence...')
        # get household size distribution
        hh_size_dist = utils.get_hh_dist(pars.datadir, pars.country.upper())
        # create households to match population size
        hh_size_arr = utils.create_multinom_rv_arr(hh_size_dist, np.arange(len(hh_size_dist), dtype=np.int32)+1, self.N)
        # populate household in grid by population density
        hh_i_coords, hh_j_coords = utils.setup_entity_coords(hh_size_arr.size, self.popgrid)
        # assign individuials to household and in turn, grid coordinates
        # initialize household IDs
        self.people.household = np.zeros(self.N, dtype=np.int32) - 1
        # initialize pop_arr and household ID arr
        pop_arr = np.arange(self.N, dtype=np.int32)
        hh_idx_arr = np.arange(hh_size_arr.size, dtype=np.int32)
        # at least one person who is of prime age (assumed to be 20) to be head of household
        head_of_household = np.random.choice(pop_arr[self.people.age>=20], size=hh_size_arr.size, replace=False)
        # remember the number of individuals filled for each household
        hh_filled_n_arr = np.ones(hh_size_arr.size, dtype=np.int32)
        # assign household to head of households
        self.people.household[head_of_household] = hh_idx_arr
        # compute the number of remaining individuals to fill for each household
        hh_unfilled_n_arr = hh_size_arr - hh_filled_n_arr
        # while we still have household with unknown individuals
        while hh_unfilled_n_arr.sum() > 0:
            # identify incompletely filled households
            mask = hh_unfilled_n_arr > 0
            hh_to_add = hh_idx_arr[mask]
            # randomly select individuals who do not have a household yet and add to these households
            household_members_to_add = np.random.choice(pop_arr[self.people.household<0], size=len(hh_to_add), replace=False)
            # assign households
            self.people.household[household_members_to_add] = hh_to_add
            # save number of individuals filled for these households
            hh_filled_n_arr[hh_to_add] += 1
            # compute the number of remaining individuals to fill for each household
            hh_unfilled_n_arr = hh_size_arr - hh_filled_n_arr
        # assign grid coordinates to individuals based on households
        self.people.location = np.zeros((self.N, 2), dtype=np.int32)
        self.people.location[:,0] = hh_i_coords[self.people.household]
        self.people.location[:,1] = hh_j_coords[self.people.household]
        # recalculate popgrid to match people's location
        self.popgrid = utils.recalculate_popgrid(self.popgrid, self.people.location)

        if pars.verbose > 0:
            print ('...done.')

    def setup_age(self, pars):
        """
        Setup age of individuals
        """
        # get age distribution
        age_dist = utils.get_age_dist(pars.datadir, pars.country.upper())
        p_arr = np.repeat(age_dist, 5)
        p_arr /= p_arr.sum()
        self.people.age = np.random.choice(np.arange(age_dist.size*5, dtype=np.int32), size=self.N, replace=True, p=p_arr)
        # 5-year bin of age
        self.people.agebin = np.floor(self.people.age/5).astype(np.int32)
