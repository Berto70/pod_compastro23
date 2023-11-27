//
// Created by lex on 2/27/23.
//

#ifndef TSUNAMI_TSUNAMI_HPP
#define TSUNAMI_TSUNAMI_HPP

#include <array>
#include <iomanip>
#include "config.hpp"
#include "chain.hpp"
#include "custom_types.hpp"
#include "classification.h"
#include "Nbodyalgorithms.hpp"
#include "leapfrog_stepped.hpp"
#include "bulirsch.hpp"
#include "keplerutils.h"
#include "simprof.hpp"

template<bool profiling, bool debug>
class TsunamiClass {
public:
    /**
     * Mass and distance unit in MSun and au.
     * Defaults to MSun and a.u.
     */
    TsunamiClass() : TsunamiClass(1, 1) {}

    TsunamiClass(double Mscale, double Lscale) :
                 System(Conf),
                 Leapfrog(System),
                 BSExtra(System, Leapfrog, tolerance) {
        std::cout << std::scientific << std::setprecision(15);

        set_units(Mscale, Lscale);
    }

    void set_units(double Mscale, double Lscale) {
        Conf.Lscale = Lscale;
        Conf.Mscale = Mscale;
        this->Mscale = Mscale;
        this->Lscale = Lscale;

        // Time unit (in yr)
        Tscale = sqrt(Lscale*Lscale*Lscale/(Mscale*KeplerUtils::G_yr_msun_au));

        // Velocity unit (in km/s)
        Vscale = Lscale/Tscale * KeplerUtils::au2km / KeplerUtils::yr2sec;
        speed_of_light = KeplerUtils::c_ms * 1e-3 / Vscale;

        System.init_postnewtonians(Mscale, Lscale);
        Marx.set_units(Mscale, Lscale);
    }

    void allocate_arrays() {
        System.Npart = N;
        System.Nchain = N - 1;
        System.allocate_arrays();
    }

    void initialize_particle_tides() {
        IO::read_tidal_table(Marx.tidetable, tidefile);
        Marx.initialize_pinfo(System.xdata, System.radius, System.mass, N);
    }

    void initialize_regularization(double alpha, double beta, double gamma) {
        Conf.alpha = alpha;
        Conf.beta = beta;
        Conf.gamma = gamma;
        Conf.TTL = (beta != 0);
        System.init_regularization();
    }

    void initialize_chain() {
        System.init_chain();
        System.initialize_chain_arrays();
    }

    void initialize_integrator() {
        Leapfrog.initialize();
        BSExtra.initialize();
        regularization_switch();
        System.init_tdetracker();
    }

    void do_step_leapfrog() {
        Leapfrog.save_step_zero(System.ch_pos, System.ch_vel, System.ch_mass,
                                System.ch_spin, System.ch_xdata);

        System.time0 = time;
        Leapfrog.integrate(nstep, timestep, dtphysical, System.ch_pos,
                           System.ch_vel, System.ch_spin);
        Leapfrog.B0 = Leapfrog.B;
        Leapfrog.omega0 = Leapfrog.omega;
    }

    void do_step_bulirsh() {
        System.time0 = time;

        if constexpr (profiling) ProfBS.start();
        BSExtra.bs_iterate(dtphysical, timestep);
        if constexpr (profiling) ProfBS.stop_store();

        Leapfrog.B0 = BSExtra.FinalTable.B;
        Leapfrog.omega0 = BSExtra.FinalTable.omega;
    }

    void regularization_switch() {
        // Using last Leapfrog integration as a proxy
        if constexpr (debug) {
            std::cout << " == REG SWITCH == " << std::endl;
            std::cout << "    B = " << Leapfrog.B << std::endl;
            std::cout << "    T = " << Leapfrog.T << std::endl;
            std::cout << "    U = " << Leapfrog.B + Leapfrog.T << std::endl;
            std::cout << "    U/T = " << (Leapfrog.B + Leapfrog.T)/Leapfrog.T << std::endl;
        }
        if (Leapfrog.B < (Conf.Usafe - 1.0) * Leapfrog.T) {
            if (Conf.gamma == 0.0) {
                if constexpr (debug) {
                    std::cout << "    U < " << Conf.Usafe << " T " << std::endl;
                    std::cout << "Setting gamma = 1 at T = " << time << std::endl;
                }
                switch_list.emplace_back(time, 1.0);
                Conf.gamma = 1.0;
            }
        } else {
            if (Conf.gamma == 1.0) {
                if constexpr (debug) {
                    std::cout << "    U > " << Conf.Usafe << " T " << std::endl;
                    std::cout << "Setting gamma = 0 at T = " << time << std::endl;
                }
                switch_list.emplace_back(time, 0.0);
                Conf.gamma = 0.0;
            }
        }
    }

    void update_time_coordinates_chain() {
        if constexpr (profiling) ProfChain.start();

        time += dtphysical;
        Eintegrator = Leapfrog.B0;
        regularization_switch();

        System.update_from_chain_to_com();

        System.find_chain();

        if (System.chain_has_changed()) {
            System.to_new_chain();
        }
        if constexpr (profiling) ProfChain.stop_store();
    }

    bool check_collision() const {
        return System.CollInfo.collflag;
    }

    /*[[nodiscard]] std::tuple<size_t,size_t> get_collision_indices() const {
        return {System.CollInfo.collind[0], System.CollInfo.collind[1]};
    }*/// Maybe when SWIG will support tuples

    void get_collision_indices(size_t &id1, size_t &id2) const {
        id1 = System.CollInfo.collind[0];
        id2 = System.CollInfo.collind[1];
    }

    void iterate_to_collision_leapfrog(double coll_thr = 1e-10) {
        double colstep_ds = System.CollInfo.colltime_ds;
        double current_ds = 0.5 * (colstep_ds + timestep);
        double deltaT = dtphysical - System.CollInfo.colltime;
        std::cout << "\n colltime " << System.CollInfo.colltime << " deltaT " << deltaT
                  << "\ncurrent_ds = " << current_ds << "\ncolstep_ds " << colstep_ds << std::endl;
        size_t nits = 0;
        while (deltaT > coll_thr) {
            Leapfrog.revert_step(System.ch_pos, System.ch_vel, System.ch_mass,
                                 System.ch_spin, System.ch_xdata);

            System.time0 = time;
            Leapfrog.integrate(nstep, current_ds, dtphysical, System.ch_pos,
                               System.ch_vel, System.ch_spin);

            if (not System.CollInfo.collflag) {
                // May happen if the collision moves around too much, let's slowly increase the timestep
                current_ds = 0.33 * (2 * colstep_ds + current_ds);
            } else {
                deltaT = dtphysical - System.CollInfo.colltime;
                colstep_ds = System.CollInfo.colltime_ds;
                current_ds = 0.5 * (colstep_ds + current_ds);
            }
            std::cout << "\nnew colltime " << System.CollInfo.colltime << " deltaT " << deltaT
                      << "\ncurrent_ds = " << current_ds << "\ncolstep_ds " << colstep_ds << std::endl;
            nits++;
            if (nits > 10000) {
                throw TsuError("Too many iterations\ncurrent_ds = " + n2s(current_ds)
                               + "\ncolstep_ds = " + n2s(colstep_ds) + "\ndeltaT = " + n2s(deltaT));
            }
        }
        // Update after converged
        Leapfrog.B0 = Leapfrog.B;
        Leapfrog.omega0 = Leapfrog.omega;
        update_time_coordinates_chain();
    }

    void iterate_to_collision_bulirsh(double coll_thr = 1e-10) {
        double colstep_ds = System.CollInfo.colltime_ds;
        double current_ds = 0.5 * (colstep_ds + timestep);
        double deltaT = dtphysical - System.CollInfo.colltime;
        std::cout << "\n colltime " << System.CollInfo.colltime << " deltaT " << deltaT
                  << "\ncurrent_ds = " << current_ds << "\ncolstep_ds " << colstep_ds << std::endl;
        size_t nits = 0;
        while (deltaT > coll_thr) {
            Leapfrog.revert_step(System.ch_pos, System.ch_vel, System.ch_mass,
                                 System.ch_spin, System.ch_xdata);

            System.time0 = time;
            BSExtra.bs_iterate(dtphysical, timestep);

            if (not System.CollInfo.collflag) {
                // May happen if the collision moves around too much, let's slowly increase the timestep
                current_ds = 0.33 * (2 * colstep_ds + current_ds);
            } else {
                deltaT = dtphysical - System.CollInfo.colltime;
                colstep_ds = System.CollInfo.colltime_ds;
                current_ds = 0.5 * (colstep_ds + current_ds);
            }
            std::cout << "\nnew colltime " << System.CollInfo.colltime << " deltaT " << deltaT
                      << "\ncurrent_ds = " << current_ds << "\ncolstep_ds " << colstep_ds << std::endl;
            nits++;
            if (nits > 10000) {
                throw TsuError("Too many iterations\ncurrent_ds = " + n2s(current_ds)
                               + "\ncolstep_ds = " + n2s(colstep_ds) + "\ndeltaT = " + n2s(deltaT));
            }
        }
        Leapfrog.B0 = BSExtra.FinalTable.B;
        Leapfrog.omega0 = BSExtra.FinalTable.omega;
        update_time_coordinates_chain();
    }

    void print_profiling() {
        if constexpr (not profiling) {
            throw TsuError("Profiling has been disabled, compile with cmake option -Dprofile=on");
        }
        ProfBS.print_avg("BS");
        double BSonly = ProfBS.avg - Leapfrog.ProfLeap.avg;
        std::cout << "BS only: " << BSonly << std::endl;
        Leapfrog.ProfLeap.print_avg("Leap");
        ProfChain.print_avg("Chain");
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///       __  _   _ ___ _  _  __  _  _    ____ _  _ _  _  ___ ___ _  __  _  _  ___                               ///
    ///      |__]  \_/   |  |__| |  | |\ |    |___ |  | |\ | |     |  | |  | |\ | [__                                ///
    ///      |      |    |  |  | |__| | \|    |    |__| | \| |___  |  | |__| | \| ___]                               ///
    ///                                                                                                              ///
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void add_particle_set(double *pos_in, size_t npos, size_t pncoord,
                          double *vel_in, size_t nvel, size_t vncoord,
                          double *mass_in, size_t nmass,
                          double *rad_in, size_t nrad,
                          long *stype_in, size_t nstype) {

        if((nmass != npos) or (nmass != nvel) or (nmass != nstype) or (nmass != nrad)) {
            throw TsuError("Provided arrays have inconsistent length");
        }
        if((pncoord != 3) or (vncoord != 3)) {
            throw TsuError("Provide 3 coordinates for each particle position and velocity");
        }

        size_t Npart = nmass;
        if(Npart < 2) {
            throw TsuError("Provide at least two particle");
        }

        for(size_t j = 0; j < Npart; j++) {
            if(std::isnan(mass_in[j]) or std::isnan(rad_in[j]) or std::isnan(stype_in[j])) {
                throw TsuError("NaN values in input (mass or radius or type)");
            }
            for(size_t i = 0; i < 3; i++) {
                if(std::isnan(pos_in[i+j*3]) or std::isnan(vel_in[i+j*3])) {
                    throw TsuError("NaN values in input (position or velocity)");
                }
            }
        }

        reallocate_arrays(Npart);

        if (TsunamiConfig::wSpins) {
            if (w_aps) {
                std::cerr << "TsunamiWarning: code was compiled with spins, but no spins are provided in add_particle_set. "
                         "Defaulting spins to zero" << std::endl;
                w_aps = false;
            }
            for(size_t i = 0; i < N; i++) {
                System.spin[i] = {0.0, 0.0, 0.0};
            }
        }

        for(size_t i = 0; i < N; i++) {
            System.pos[i].x = pos_in[3*i];
            System.pos[i].y = pos_in[3*i+1];
            System.pos[i].z = pos_in[3*i+2];
            System.vel[i].x = vel_in[3*i];
            System.vel[i].y = vel_in[3*i+1];
            System.vel[i].z = vel_in[3*i+2];
            System.mass[i] = mass_in[i];
            System.radius[i] = rad_in[i];
            System.xdata[i].stype = static_cast<ptype>(stype_in[i]);
        }
        Nbodyalgorithms::scale_to_cdm(System.pos, System.vel, System.mass, N);

        initialize_regularization(Conf.alpha, Conf.beta, Conf.gamma);
        initialize_chain();
        initialize_integrator();
    }

    void add_particle_set(double *pos_in, size_t npos, size_t pncoord,
                          double *vel_in, size_t nvel, size_t vncoord,
                          double *mass_in, size_t nmass,
                          double *rad_in, size_t nrad,
                          long *stype_in, size_t nstype,
                          double *spin_in, size_t nspin, size_t sncoord) {

        if (not TsunamiConfig::wSpins) {
            if (w_aps) {
                std::cerr << "TsunamiWarning: code was not compiled with spins, ignoring the extra spin argument"
                          << std::endl;
                w_aps = false;
            }
            add_particle_set(pos_in, npos, pncoord, vel_in, nvel, vncoord, mass_in, nmass, rad_in, nrad, stype_in, nstype);
            return;
        }

        if((nmass != npos) or (nmass != nvel) or (nmass != nstype) or (nmass != nrad) or (nmass != nspin)) {
            throw TsuError("Provided arrays have inconsistent length");
        }
        if((pncoord != 3) or (vncoord != 3) or (sncoord != 3)) {
            throw TsuError("Provide 3 coordinates for each particle position and velocity");
        }

        size_t Npart = nmass;
        if(Npart < 2) {
            throw TsuError("Provide at least two particle");
        }

        for(size_t j = 0; j < Npart; j++) {
            if(std::isnan(mass_in[j]) or std::isnan(rad_in[j]) or std::isnan(stype_in[j])) {
                throw TsuError("NaN values in input (mass or radius or type)");
            }
            for(size_t i = 0; i < 3; i++) {
                if(std::isnan(pos_in[i+j*3]) or std::isnan(vel_in[i+j*3]) or std::isnan(spin_in[i+j*3])) {
                    throw TsuError("NaN values in input (position or velocity or spin)");
                }
            }
        }

        reallocate_arrays(Npart);

        for(size_t i = 0; i < N; i++) {
            System.pos[i].x = pos_in[3*i];
            System.pos[i].y = pos_in[3*i+1];
            System.pos[i].z = pos_in[3*i+2];
            System.vel[i].x = vel_in[3*i];
            System.vel[i].y = vel_in[3*i+1];
            System.vel[i].z = vel_in[3*i+2];
            System.spin[i] = {spin_in[3*i], spin_in[3*i+1], spin_in[3*i+2]};
            System.mass[i] = mass_in[i];
            System.radius[i] = rad_in[i];
            System.xdata[i].stype = static_cast<ptype>(stype_in[i]);
        }
        Nbodyalgorithms::scale_to_cdm(System.pos, System.vel, System.mass, N);

        initialize_regularization(Conf.alpha, Conf.beta, Conf.gamma);
        initialize_chain();
        initialize_integrator();
    }


    void sync_internal_state(double *pos_inout, size_t npos, size_t pncoord,
                             double *vel_inout, size_t nvel, size_t vncoord) {

        if((N != npos) or  (N != nvel)) {
            throw TsuError("Provided arrays have lengths different from particle number");
        }
        if((pncoord != 3) or (vncoord != 3)) {
            throw TsuError("Provide 3 coordinates for each particle");
        };

        for(size_t i = 0; i < N; i++) {
            pos_inout[3*i] = System.pos[i].x;
            pos_inout[3*i+1] = System.pos[i].y;
            pos_inout[3*i+2] = System.pos[i].z;
            vel_inout[3*i] = System.vel[i].x;
            vel_inout[3*i+1] = System.vel[i].y;
            vel_inout[3*i+2] = System.vel[i].z;
        }
    }

    void sync_internal_state(double *pos_inout, size_t npos, size_t pncoord,
                             double *vel_inout, size_t nvel, size_t vncoord,
                             double *spin_inout, size_t nspin, size_t sncoord) {
        if (not TsunamiConfig::wSpins) {
            if (w_sis) {
                std::cerr << "TsunamiWarning: code was not compiled with spins, ignoring the extra spin argument" << std::endl;
                w_sis = false;
            }
            sync_internal_state(pos_inout, npos, pncoord, vel_inout, nvel, vncoord);
            return;
        }

        if((N != npos) or  (N != nvel) or (N != nspin)) {
            throw TsuError("Provided arrays have lengths different from particle number");
        }
        if((pncoord != 3) or (vncoord != 3) or (sncoord != 3)) {
            throw TsuError("Provide 3 coordinates for each particle");
        };

        for(size_t i = 0; i < N; i++) {
            pos_inout[3*i] = System.pos[i].x;
            pos_inout[3*i+1] = System.pos[i].y;
            pos_inout[3*i+2] = System.pos[i].z;
            vel_inout[3*i] = System.vel[i].x;
            vel_inout[3*i+1] = System.vel[i].y;
            vel_inout[3*i+2] = System.vel[i].z;
            spin_inout[3*i] = System.spin[i].x;
            spin_inout[3*i+1] = System.spin[i].y;
            spin_inout[3*i+2] = System.spin[i].z;
        }
    }

    void override_masses(double *mass_in, size_t nmass) {
        if(N != nmass) {
            throw TsuError("Provided arrays have lengths different from particle number");
        }

        for(size_t i = 0; i < N; i++) {
            System.mass[i] = mass_in[i];
        }

        // Different from reset_integrator_sameN because we do not reinitialize the chain
        Nbodyalgorithms::scale_to_cdm(System.pos, System.vel, System.mass, N);
        energy = Nbodyalgorithms::energy_calculation(System.pos, System.vel, System.mass, N, pot, kin);
        System.update_chained_data(); // Update chained data
        BSExtra.reset_bulirsch(); 	// Reset BS and its internal leapfrog
    }

    /**
     * Overwrites the stored positions and velocities with the provided ones, leaving unchanged the number of particles
     * If changing also masses, *update masses first*
     * @param pos_in
     * @param npos
     * @param pncoord
     * @param vel_in
     * @param nvel
     * @param vncoord
     */
    void override_position_and_velocities(double *pos_in, size_t npos, size_t pncoord,
                                          double *vel_in, size_t nvel, size_t vncoord) {
        if((N != npos) or  (N != nvel)) {
            throw TsuError("Provided arrays have lengths different from particle number");
        }
        if((pncoord != 3) or (vncoord != 3)) {
            throw TsuError("Provide 3 coordinates for each particle");
        };

        for(size_t i = 0; i < N; i++) {
            System.pos[i].x = pos_in[3*i];
            System.pos[i].y = pos_in[3*i+1];
            System.pos[i].z = pos_in[3*i+2];
            System.vel[i].x = vel_in[3*i];
            System.vel[i].y = vel_in[3*i+1];
            System.vel[i].z = vel_in[3*i+2];
        }

        Nbodyalgorithms::scale_to_cdm(System.pos, System.vel, System.mass, N);
        energy = Nbodyalgorithms::energy_calculation(System.pos, System.vel, System.mass, N, pot, kin);
        System.find_chain();
        System.initialize_chain_arrays(); // Redo chain
        BSExtra.reset_bulirsch(); 	// Reset BS and its internal leapfrog
    }

    /**
     * Initializes tidal parameters. Time lag is given in N-body units
     * @param kaps_in
     * @param nkaps
     * @param taulag_in
     * @param ntaulag
     * @param polyt_in
     * @param npolyt
     */
    void initialize_tidal_parameters(double *kaps_in, size_t nkaps,
                                     double *taulag_in, size_t ntaulag,
                                     double *polyt_in, size_t npolyt) {
        if (TsunamiConfig::wSpins) {
            throw TsuError("Code was compiled with spins\nadd gyration radii as last argument");
        }

        if((N != nkaps) or (N != ntaulag) or (N != npolyt)) {
            throw TsuError("Provided arrays have lengths different from particle number");
        }

        for(size_t i=0; i < N; i++) {
            if((kaps_in[i] <= 0) or (taulag_in[i] <= 0) or (polyt_in[i] <= 0)) {
                System.xdata[i].hastide = false;
                //std::cerr<<"Particle "<<i<<" has a tidal parameter less or equal than zero: disabling tides on this particle"<<std::endl;
            } else {
                System.xdata[i].hastide = true;
                System.xdata[i].polyt = polyt_in[i];
                System.xdata[i].kaps = kaps_in[i];
                System.xdata[i].taulag = taulag_in[i];

                double Qt = 2 * System.xdata[i].kaps / (1 + 2 * System.xdata[i].kaps);
                double R5 = pow(System.radius[i], 5);

                // True parameters used
                System.xdata[i].Atide = R5 * Qt / (1 - Qt);
                System.xdata[i].sigmadiss = 2.0 * R5 / (System.xdata[i].Atide*System.xdata[i].Atide) * System.xdata[i].kaps * System.xdata[i].taulag / 3;
            }
        }

        System.update_chained_data();
    }

    /**
     * Initializes tidal parameters. Time lag is given in N-body units
     * @param kaps_in
     * @param nkaps
     * @param taulag_in
     * @param ntaulag
     * @param polyt_in
     * @param npolyt
     */
    void initialize_tidal_parameters(double *kaps_in, size_t nkaps,
                                     double *taulag_in, size_t ntaulag,
                                     double *polyt_in, size_t npolyt,
                                     double *gyrad_in, size_t ngyrad) {

        if((N != nkaps) or (N != ntaulag) or (N != npolyt) or (N != ngyrad)) {
            throw TsuError("Provided arrays have lengths different from particle number");
        }

        for(size_t i=0; i < N; i++) {
            if(((kaps_in[i] <= 0) or (taulag_in[i] <= 0)) and (polyt_in[i] <= 0)) {
                System.xdata[i].hastide = false;
                //std::cerr<<"Particle "<<i<<" has a tidal parameter less or equal than zero: disabling tides on this particle"<<std::endl;
            } else {
                System.xdata[i].hastide = true;
                System.xdata[i].polyt = polyt_in[i];
                System.xdata[i].kaps = kaps_in[i];
                System.xdata[i].taulag = taulag_in[i];

                double gyrad = gyrad_in[i] * System.radius[i];
                System.xdata[i].inert = gyrad*gyrad * System.mass[i];

                double Qt = 2 * System.xdata[i].kaps / (1 + 2 * System.xdata[i].kaps);
                double R5 = pow(System.radius[i], 5);

                // True parameters used
                System.xdata[i].Atide = R5 * Qt / (1 - Qt);
                System.xdata[i].sigmadiss = 2.0 * R5 / (System.xdata[i].Atide*System.xdata[i].Atide) * System.xdata[i].kaps * System.xdata[i].taulag / 3;
            }
        }

        System.update_chained_data();
    }


/**
 * Evolves the system to the given final time
 * @param[in] tfin
 */
    void evolve_system(double tfin) {
        // All sort of checks
        if(tfin <= time)
            throw TsuError("Trying to evolve a system that has already reached tfin");
        if (System.CollInfo.collflag)
            throw TsuError("Trying to evolve a system that has already collided\nsolve the collision and resume evolution");

        while (not stopcond) { // Start loop

            // Advance system
            try {
                do_step_bulirsh();
            } catch (TsunamiError &error) {
                std::time_t t = std::time(nullptr);
                std::tm tm = *std::localtime(&t);
                std::ostringstream oss;
                oss << std::put_time(&tm, "%d-%m-%Y-%H-%M-%S");
                string fname = "emergency_crash_" + oss.str() + ".bin";
                BSExtra.dt = BSExtra.dt_prev; // If dt screwed up, save the previous one
                save_restart_file(fname);
                throw TsuError("Saved " + fname + ", now ship the file to Alessandro");
            }

#ifdef STOPCOND
            stopcond = StopLogger->check_stopping_condition(pos, vel, mass, rad, ctime);
#endif
            if (check_collision()) {
                //iterate_to_collision_leapfrog();
                //Tsunami.revert_step();
                break;
            }

            update_time_coordinates_chain();

            if(time >= tfin) { // If reached target time, break out
                energy = Nbodyalgorithms::energy_calculation(System.pos, System.vel,
                                                             System.mass, N,
                                                             pot, kin);
                deltaE = fabs(log(fabs((kin + (Eintegrator))/(pot))));
                break;
            }
        } // End loop
    }


/**
 * Evolves the system for a single step, making sure it does not integrate beyond tfin
 * @param[in] tfin
 */
    void evolve_system_dtmax(double tfin) {
        // All sort of checks
        if(tfin <= time)
            throw TsuError("Trying to evolve a system that has already reached tfin");
        if (System.CollInfo.collflag)
            throw TsuError("Trying to evolve a system that has already collided\nsolve the collision and resume evolution");

        double dtime = tfin - time;
        double timestep_dt = estimate_ds_from_dt(dtime);
        timestep = (timestep_dt > timestep) ? timestep : timestep_dt;

        do_step_bulirsh();

#ifdef STOPCOND
            stopcond = StopLogger->check_stopping_condition(pos, vel, mass, rad, ctime);
#endif
        check_collision();
        update_time_coordinates_chain();

        energy = Nbodyalgorithms::energy_calculation(System.pos, System.vel,
                                                         System.mass, N,
                                                         pot, kin);
        deltaE = fabs(log(fabs((kin + (Eintegrator))/(pot))));
    }

    void commit_parameters() {
        Leapfrog.update_parameters();
    }

    void reallocate_arrays(size_t Npart) {
        if (N == 0 or N != Npart) {
            N = Npart;
            if (System.Npart > 0) System.deallocate_arrays();
            System.Npart = Npart;
            System.Nchain = Npart - 1;
            System.allocate_arrays();
            System.chain.resize(Npart);
        }
    }

    void revert_step() {
        Leapfrog.revert_step(System.ch_pos, System.ch_vel, System.ch_mass, System.ch_spin, System.ch_xdata);
    }

    void get_chain_vectors(double **chpos_out, int *npos, int *pncoord,
                           double **chvel_out, int *nvel, int *vncoord) {

        if (System.Npart < 2) {
            throw TsuError("No particles found");
        }

        *vncoord = *pncoord = 3;
        *npos = *nvel = System.Nchain;
        auto *tmp_pos = new double [3*System.Nchain];
        auto *tmp_vel = new double [3*System.Nchain];

        for (size_t i=0; i<System.Nchain; i++) {
            tmp_pos[i*System.Nchain + 0] = System.ch_pos[i].x;
            tmp_pos[i*System.Nchain + 1] = System.ch_pos[i].y;
            tmp_pos[i*System.Nchain + 2] = System.ch_pos[i].z;
            tmp_vel[i*System.Nchain + 0] = System.ch_vel[i].x;
            tmp_vel[i*System.Nchain + 1] = System.ch_vel[i].y;
            tmp_vel[i*System.Nchain + 2] = System.ch_vel[i].z;
        }
        *chpos_out = tmp_pos;
        *chvel_out = tmp_vel;
    }


    void sync_masses(double *mass_out, size_t nmass) const {
        if (System.Npart < 2) {
            throw TsuError("No particles found");
        }
        if(System.Npart != nmass) {
            throw TsuError("Provided arrays have lengths different from particle number");
        }

        for (size_t i=0; i<N; i++) {
            mass_out[i] = System.mass[i];
        }
    }

    void sync_radii(double *rad_out, size_t nrad) const {
        if (System.Npart < 2) {
            throw TsuError("No particles found");
        }

        if(System.Npart != nrad) {
            throw TsuError("Provided arrays have lengths different from particle number");
        }

        for (size_t i=0; i<System.Npart; i++) {
            rad_out[i] = System.radius[i];
        }
    }

    void sync_eloss(double *eloss_out, size_t neloss) const {
        if (System.Npart < 2) {
            throw TsuError("No particles found");
        }

        if(System.Npart != neloss) {
            throw TsuError("Provided arrays have lengths different from particle number");
        }

        for (size_t i=0; i<System.Npart; i++) {
            eloss_out[i] = System.xdata[i].eloss;
        }
    }

    double estimate_ds_from_dt(const double dt) {
        double dt_u = Leapfrog.calc_dt_pos(1.0, Leapfrog.T + Leapfrog.B, Leapfrog.omega);
        return dt/dt_u;
    }

    Nbodyalgorithms::PTDELog & get_PTDELog() {
        if constexpr (TsunamiConfig::useTDEtracker) {
        } else {
            std::cerr << "TsunamiWarning: code was not compiled with tdetracker, but user is requesting PDTELog\n\trecompile with -Dtdetracker=on" << std::endl;
        }
        return System.PTDEInfo;
    }

    void save_restart_file(const std::string& fname) {
        if (N < 1) throw TsuError("No particles found");

        std::ofstream outfile(fname, std::ios::binary | std::ios::out);
        if (!outfile) throw TsuError("Cannot open file " + fname);

        outfile.write(reinterpret_cast<const char *>(&N), sizeof(size_t));
        outfile.write(reinterpret_cast<const char *>(&timestep), sizeof(double));
        outfile.write(reinterpret_cast<const char *>(&dtphysical), sizeof(double));  // Not necessary but useful info
        outfile.write(reinterpret_cast<char *>(&Conf.wPNs), sizeof(bool));
        outfile.write(reinterpret_cast<char *>(&Conf.wEqTides), sizeof(bool));
        outfile.write(reinterpret_cast<char *>(&Conf.wDynTides), sizeof(bool));
        outfile.write(reinterpret_cast<char *>(&Conf.wExt), sizeof(bool));
        outfile.write(reinterpret_cast<char *>(&Conf.wExt_vdep), sizeof(bool));
        outfile.write(reinterpret_cast<char *>(&Conf.wMassEvol), sizeof(bool));
        outfile.write(reinterpret_cast<char *>(&Conf.Mscale), sizeof(double));
        outfile.write(reinterpret_cast<char *>(&Conf.Lscale), sizeof(double));
        outfile.write(reinterpret_cast<char *>(&Conf.alpha), sizeof(double));
        outfile.write(reinterpret_cast<char *>(&Conf.beta), sizeof(double));
        outfile.write(reinterpret_cast<char *>(&Conf.gamma), sizeof(double));
        outfile.write(reinterpret_cast<char *>(&Conf.Usafe), sizeof(double));
        outfile.write(reinterpret_cast<char *>(&Conf.dcoll), sizeof(double));
        outfile.write(reinterpret_cast<const char *>(&tolerance), sizeof(double));
        outfile.write(reinterpret_cast<const char *>(&BSExtra.kopt_now), sizeof(size_t));
        outfile.write(reinterpret_cast<const char *>(&Leapfrog.B0), sizeof(double));
        outfile.write(reinterpret_cast<const char *>(&Leapfrog.omega0), sizeof(double));
        outfile.write(reinterpret_cast<const char *>(&deltaE), sizeof(double));
        outfile.write(reinterpret_cast<const char *>(&energy), sizeof(double));
        outfile.write(reinterpret_cast<const char *>(&time), sizeof(double));
        outfile.write(reinterpret_cast<const char *>(System.chain.data()), System.Npart*sizeof(size_t));
        outfile.write(reinterpret_cast<const char *>(System.ch_pos), System.Nchain*sizeof(double3));
        outfile.write(reinterpret_cast<const char *>(System.ch_vel), System.Nchain*sizeof(double3));
        outfile.write(reinterpret_cast<const char *>(System.ch_mass), System.Npart*sizeof(double));
        outfile.write(reinterpret_cast<const char *>(System.ch_radius), System.Npart*sizeof(double));
        outfile.write(reinterpret_cast<const char *>(System.ch_xdata), System.Npart*sizeof(pinfo));

        std::vector<double3> zerospin; // If spin is not included
        double3 *spin_data_pointer;
        if (TsunamiConfig::wSpins) {
            spin_data_pointer = System.ch_spin;
        } else {
            for (size_t i=0; i<N; i++) {
                zerospin.emplace_back(0.0, 0.0, 0.0);
            }
            spin_data_pointer = zerospin.data();
        }
        outfile.write(reinterpret_cast<const char *>(spin_data_pointer), System.Npart*sizeof(double3));

        outfile.close();
        if(!outfile.good()) throw TsuError("Error occurred during writing of " + fname);


        /*std::cout << N << std::endl;
        System.print_chain();
        for(size_t i=0; i<System.Npart; i++) std::cout << System.mass[i] << " ";
        std::cout << std::endl;
        for(size_t i=0; i<System.Npart; i++) std::cout << System.radius[i] << " ";
        std::cout << std::endl;
        for(size_t i=0; i<System.Npart; i++) std::cout << System.xdata[i] << std::endl;
        std::cout << std::endl;
        for(size_t i=0; i<System.Npart; i++) std::cout << System.pos[i] << " ";
        std::cout << std::endl;
        for(size_t i=0; i<System.Npart; i++) std::cout << System.vel[i] << " ";
        std::cout << std::endl;
        for(size_t i=0; i<System.Npart; i++) std::cout << System.ch_pos[i] << " ";
        std::cout << std::endl;
        for(size_t i=0; i<System.Npart; i++) std::cout << System.ch_vel[i] << " ";
        std::cout << std::endl;
        std::cout << "energy " << energy << std::endl;*/
    }


    void load_restart_file(const std::string& fname) {
        std::ifstream infile(fname, std::ios::binary | std::ios::in);
        if (!infile) throw TsuError("Cannot open file " + fname);


        size_t Npart;
        infile.read(reinterpret_cast<char *>(&Npart), sizeof(size_t));
        if(Npart < 2) throw TsuError("Cannot initialize integrator with less than two particles");

        reallocate_arrays(Npart);

        ch_real B0, omega0;
        size_t k_opt;
        infile.read(reinterpret_cast<char *>(&timestep), sizeof(double));
        infile.read(reinterpret_cast<char *>(&dtphysical), sizeof(double));  // Not necessary but useful info
        infile.read(reinterpret_cast<char *>(&Conf.wPNs), sizeof(bool));
        infile.read(reinterpret_cast<char *>(&Conf.wEqTides), sizeof(bool));
        infile.read(reinterpret_cast<char *>(&Conf.wDynTides), sizeof(bool));
        infile.read(reinterpret_cast<char *>(&Conf.wExt), sizeof(bool));
        infile.read(reinterpret_cast<char *>(&Conf.wExt_vdep), sizeof(bool));
        infile.read(reinterpret_cast<char *>(&Conf.wMassEvol), sizeof(bool));
        infile.read(reinterpret_cast<char *>(&Conf.Mscale), sizeof(double));
        infile.read(reinterpret_cast<char *>(&Conf.Lscale), sizeof(double));
        infile.read(reinterpret_cast<char *>(&Conf.alpha), sizeof(double));
        infile.read(reinterpret_cast<char *>(&Conf.beta), sizeof(double));
        infile.read(reinterpret_cast<char *>(&Conf.gamma), sizeof(double));
        infile.read(reinterpret_cast<char *>(&Conf.Usafe), sizeof(double));
        infile.read(reinterpret_cast<char *>(&Conf.dcoll), sizeof(double));
        infile.read(reinterpret_cast<char *>(&tolerance), sizeof(double));
        infile.read(reinterpret_cast<char *>(&k_opt), sizeof(size_t));
        infile.read(reinterpret_cast<char *>(&B0), sizeof(double));
        infile.read(reinterpret_cast<char *>(&omega0), sizeof(double));
        infile.read(reinterpret_cast<char *>(&deltaE), sizeof(double));
        infile.read(reinterpret_cast<char *>(&energy), sizeof(double));
        infile.read(reinterpret_cast<char *>(&time), sizeof(double));
        infile.read(reinterpret_cast<char *>(System.chain.data()), System.Npart*sizeof(size_t));
        infile.read(reinterpret_cast<char *>(System.ch_pos), System.Nchain*sizeof(double3));
        infile.read(reinterpret_cast<char *>(System.ch_vel), System.Nchain*sizeof(double3));
        infile.read(reinterpret_cast<char *>(System.ch_mass), System.Npart*sizeof(double));
        infile.read(reinterpret_cast<char *>(System.ch_radius), System.Npart*sizeof(double));
        infile.read(reinterpret_cast<char *>(System.ch_xdata), System.Npart*sizeof(pinfo));

        if (TsunamiConfig::wSpins) {
            infile.read(reinterpret_cast<char *>(System.ch_spin), System.Npart*sizeof(double3));
        } else {
            std::vector<double3> zerospin(System.Npart); // If spin is not included
            infile.read(reinterpret_cast<char *>(zerospin.data()), System.Npart*sizeof(double3));
        }

        infile.close();
        if(!infile.good()) throw TsuError("Error occurred during the reading of " + fname);

        std::cout << Npart << std::endl;
        System.print_chain();

        set_units(Conf.Mscale, Conf.Lscale);

        System.copy_from_chain_to_com();
        System.init_chain();

        initialize_regularization(Conf.alpha, Conf.beta, Conf.gamma);
        initialize_integrator();

        /*for(size_t i=0; i<System.Npart; i++) std::cout << System.mass[i] << " ";
        std::cout << std::endl;
        for(size_t i=0; i<System.Npart; i++) std::cout << System.radius[i] << " ";
        std::cout << std::endl;
        for(size_t i=0; i<System.Npart; i++) std::cout << System.xdata[i] << std::endl;
        std::cout << std::endl;
        for(size_t i=0; i<System.Npart; i++) std::cout << System.pos[i] << " ";
        std::cout << std::endl;
        for(size_t i=0; i<System.Npart; i++) std::cout << System.vel[i] << " ";
        std::cout << std::endl;
        for(size_t i=0; i<System.Npart; i++) std::cout << System.ch_pos[i] << " ";
        std::cout << std::endl;
        for(size_t i=0; i<System.Npart; i++) std::cout << System.ch_vel[i] << " ";
        std::cout << std::endl;
        energy = Nbodyalgorithms::energy_calculation(System.pos, System.vel, System.mass, N, pot, kin);
        std::cout << "energy " << energy << std::endl;*/

        BSExtra.kopt_now = k_opt;
        BSExtra.FinalTable.B = Leapfrog.B0 = B0;
        BSExtra.FinalTable.omega = Leapfrog.omega0 = omega0;
    }


    std::vector<std::pair<double, double>> switch_list;

    /// Number of particles
    size_t N = 0;

    /// Paths and default names
    // Current path parameter file
    string exepath = IO::get_exe_path();
    string paramfile_locpath = "input/tsunami_parameters.txt";
    string fname_locpath = "input/tsunami_default_input.dat";
    string tidetile_locpath = "input/tsunami_tide_table.dat";

    string paramfile = exepath + paramfile_locpath;
    string fname_in = exepath + fname_locpath;
    string tidefile = exepath + tidetile_locpath;
    string out_name = "output.dat";
    string ene_name = "energy.dat";
    string coll_name = "collision.dat";

    /// Parameters
    double timestep = 1.e-13;
    double tolerance = 1.0e-13;
    TsunamiConfig Conf;

    /// Sub-classes
    ChainSys System;
    Classification Marx;
    LeapfrogStepped Leapfrog;
    BSExtrap BSExtra;
    size_t nstep = 2;

    // Useful unit scales
    double Mscale; ///< Mass scale (in Msun)
    double Lscale; ///< Length scale (in parsec)
    double Tscale; ///< Time scale (in yr)
    double Vscale; ///< Velocity scale (in km/s)
    double speed_of_light;

    double time = 0.0;
    double energy = 0.0;
    double pot = 0.0;
    double kin = 0.0;
    double eoff = 0.0;

    bool stopcond = false;

    double deltaE = 0.0;
    double Eintegrator = 0.0;
    double dtphysical = 0.0; // Physical timestep

    // Profiling utilities
    SimProf ProfBS = SimProf();
    SimProf ProfChain = SimProf();

    // Warnings
    bool w_sis = true;
    bool w_aps = true;

};

typedef TsunamiClass<TsunamiConfig::useProfiling, (TsunamiConfig::debug_bs or TsunamiConfig::debug_lf or TsunamiConfig::debug_ch)> TsunamiCode;

#endif //TSUNAMI_TSUNAMI_HPP
