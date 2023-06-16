import neopatat
import sciris as sc
import numpy as np
import scipy.sparse as sp
import os
import sys
import argparse
sys.setrecursionlimit(10**6)

def simulate(params):
    # create simulation object
    pars = sc.objdict(
        datadir = params.datadir,
        # country to simulate
        country = params.country,
        # preload datafolder
        preload_datafolder = params.preload_datafolder,
        # simulation period
        ndays = params.ndays,
        # start date,
        startdate = params.startdate,
        # pathogen
        pathogen = params.pathogen,
        # initial number of infections
        init_n = params.init_n,
        # initial percentile of population density when starting infection occur
        init_popden_p = params.init_popden_p,
        # beta
        beta = params.beta,
        # setting beta multiplier (household, school, workplace, random)
        setting_beta_f = [1., 1., 1., 1.],
        # overdispersion pararameters
        overdispersion_bool = params.dispersion,
        overdispersion_pars = [params.oda, params.odb],

        # introduce mutant or not
        mt_bool = params.mt_bool,
        # day of mutant introduction
        mt_intro = params.mt_intro,
        # relative transmissibility of mutant
        mt_trans_f = params.mt_trans_f,
        # percentile of population density where variant was introduced/emerged
        mt_init_popden_p = params.mt_init_popden_p,
        # initial number of mutant infections
        mt_init_n = params.mt_init_n,
        # cross-immunity of wt to mt
        cross_wt_immunity = params.cross_wt_immunity,

        # likelihood that symptomatic person would isolate at home
        redtranslike = params.redtranslike,
        # reduction factor in household transmission when symptomatic person isolate
        isolate_mild_trans_red = [params.hh_red, params.sch_red, params.wrk_red, params.rnd_red],
        isolate_sev_trans_red = params.isolate_sev_trans_red,
        # commute dist pars
        commute_pars = [params.commute_par_a, params.commute_par_b],
        # number of threads
        num_threads = params.num_threads,
        # verbose
        verbose = params.verbose,
    )

    sim = neopatat.Sim(pars)
    
    # save results
    if not os.path.isdir('./results'):
        os.mkdir("./results")
    outdir = "./results/" + params.outdir + "/"
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    np.savez(outdir+"resarr.npz",
             isolation = sim.epidemic.isolation,
             infection_b_by_place = sim.epidemic.infection_b_by_place)

    sp.save_npz(outdir+"transmission_history.npz", sim.epidemic.transmission_history.tocoo())

    sp.save_npz(outdir+"variant_history.npz", sim.epidemic.variant_history.tocoo())

    sp.save_npz(outdir+"state.npz", sim.epidemic.state.tocoo())

    sp.save_npz(outdir+"symptoms.npz", sim.epidemic.symptoms.tocoo())

    return

def make_parser():
    """
    Make argument parser
    """

    parser = argparse.ArgumentParser(description='flu/rsv/covid-19 epidemic simulator')
    subparsers = parser.add_subparsers()

    # simulation
    sim_parser = subparsers.add_parser('simulate', description='run simulation')
    sim_parser.add_argument('--pathogen', type = str, default='sars-cov-2', help='pathogen')
    sim_parser.add_argument('--country', type = str, default='geo', help='ISO3 country code')
    sim_parser.add_argument('--preload_datafolder', type = str, default='./preload/georgia_20230523/', help='preload data folder path')
    sim_parser.add_argument('--outdir', type = str, default='geogia_testrun', help='output directory')

    sim_parser.add_argument('--init_n', type = int, default = 5, help='number of initial infections')
    sim_parser.add_argument('--ndays', type = int, default = 180, help='length of period to simulate in days')
    sim_parser.add_argument('--init_popden_p', type = float, default = .95, help='initial percentile of population density when starting infection occur')
    sim_parser.add_argument('--beta', type = float, default = .0331, help='probability of transmission per contact')
    sim_parser.add_argument('--startdate', type = str, default = '2021-01-01', help = 'start date of simulation')

    sim_parser.add_argument('--mt_bool', type = int, default = 0, help='introduce mutant virus')
    sim_parser.add_argument('--mt_intro', type = int, default = 30, help='day to introduce mutant')
    sim_parser.add_argument('--mt_trans_f', type = float, default = 1.5, help='relative transmissibility to wild-type')
    sim_parser.add_argument('--mt_init_popden_p', type = float, default = .95, help='initial percentile of population density when starting infection occur')
    sim_parser.add_argument('--mt_init_n', type = int, default = 1, help='number of initial infections')
    sim_parser.add_argument('--cross_wt_immunity', type = float, default = 0., help='immunity to mutant conferred by wild-type infection')

    sim_parser.add_argument('--redtranslike', type = float, default = 0.35, help='likelihood that mild symptomatic person would reduce transmission')
    sim_parser.add_argument('--hh_red', type = float, default = 0.65, help='reduction factor in transmission at household when symptomatic mild person isolate')
    sim_parser.add_argument('--sch_red', type = float, default = 1.0, help='reduction factor in transmission at school when symptomatic mild person isolate')
    sim_parser.add_argument('--wrk_red', type = float, default = 1.0, help='reduction factor in transmission at workplace when symptomatic mild person isolate')
    sim_parser.add_argument('--rnd_red', type = float, default = 0.2, help='reduction factor in transmission at random community when symptomatic mild person isolate')
    sim_parser.add_argument('--isolate_sev_trans_red', type = float, default = 1.0, help='reduction factor in transmission when symptomatic severe person isolate')

    sim_parser.add_argument('--dispersion', type = int, default = 1, help='boolean for dispersion (superspreading) effect')
    sim_parser.add_argument('--oda', type = float, default = 1., help='dispersion factor a')
    sim_parser.add_argument('--odb', type = float, default = 0.45, help='dispersion factor a')

    sim_parser.add_argument('--commute_par_a', type = float, default = 3.8, help='commute par a')
    sim_parser.add_argument('--commute_par_b', type = float, default = 2.32, help='commute par b')

    sim_parser.add_argument('--datadir', type = str, default='./data/', help='data folder path')
    sim_parser.add_argument('--num_threads', type = int, default=28, help='number of threads')
    sim_parser.add_argument('--verbose', type = int, default=1, help='verbose')
    sim_parser.set_defaults(func=simulate)

    return parser

def main():
    # parse arguments
    parser = make_parser()
    params = parser.parse_args()
    # run function
    if params == argparse.Namespace():
        parser.print_help()
        return_code = 0
    else:
        return_code = params.func(params)

    sys.exit(return_code)

if __name__ == '__main__':
    main()
