//
// Created by lex on 19/12/18.
//

#ifndef TSUNAMI_NBODYUTILS_H
#define TSUNAMI_NBODYUTILS_H

#include <cmath>
#include <map>
#include "custom_types.hpp"
#include "config.hpp"

/**
 * Nbodyalgorithms class
 */
namespace Nbodyalgorithms {

    inline void scale_to_cdm(ch_real3 *pos, ch_real3 *vel, const ch_real *mass, size_t Npart) {

        ch_real3 cmp(0);
        ch_real3 cmv(0);
        ch_real totalmass = ch_real(0);

        for (size_t i = 0; i < Npart; i++) {
            cmp += mass[i] * pos[i];
            cmv += mass[i] * vel[i];
            totalmass += mass[i];
        }

        cmp /= totalmass;
        cmv /= totalmass;

        for (size_t i = 0; i < Npart; i++) {
            pos[i] -= cmp;
            vel[i] -= cmv;
        }
    }

    inline void scale_to_cdm_single(ch_real3 *loc_pv, const ch_real *mass, size_t Npart) {

        ch_real3 cmpv(0);
        ch_real totalmass = ch_real(0);

        for (size_t i = 0; i < Npart; i++) {
            cmpv += mass[i] * loc_pv[i];
            totalmass += mass[i];
        }

        cmpv /= totalmass;

        for (size_t i = 0; i < Npart; i++) {
            loc_pv[i] -= cmpv;
        }
    }

    inline ch_real energy_calculation(const ch_real3 *pos, const ch_real3 *vel, const ch_real *mass,
                                      size_t Npart, ch_real &potential, ch_real &kinetic) {

        potential = 0.0;
        kinetic = 0.0;

        ch_real3 dp;

        for (size_t k = 0; k < Npart; k++) {
            kinetic += 0.5 * mass[k] * (vel[k] * vel[k]);

            for (size_t j = 0; j < k; j++) {
                dp = pos[j] - pos[k];
                ch_real inv_dr = 1.0 / sqrt(dp.x * dp.x + dp.y * dp.y + dp.z * dp.z);

                potential += mass[j] * mass[k] * inv_dr;
            }
        }
        //std::cout<<potential<<"  "<<kinetic<<std::endl;

        return (kinetic - potential);
    }

    inline ch_real dynamical_timescale(const ch_real3 *pos, const ch_real3 *vel, const ch_real *mass, size_t Npart) {

        ch_real potential = 0.0;
        ch_real kinetic = 0.0;
        ch_real mtot = 0.0;

        for (size_t k = 0; k < Npart; k++) {
            kinetic += 0.5 * mass[k] * (vel[k] * vel[k]);
            mtot += mass[k];

            for (size_t j = 0; j < k; j++) {
                ch_real3 dp = pos[j] - pos[k];
                ch_real inv_dr = 1.0 / sqrt(dp.x * dp.x + dp.y * dp.y + dp.z * dp.z);

                potential += mass[j] * mass[k] * inv_dr;
            }
        }
        ch_real etot = fabs(potential - kinetic);
        return 0.5 * sqrt(mtot * mtot * mtot * mtot * mtot / (etot * etot * etot));
    }

    /// Collision type
    enum CollType {
        NOCOLLISION = 0,
        COLLISION_CANDIDATE = 1,
        COLLISION_REAL = 2,
        COLLISION_TDE = 3,
    };

    /// Collision log
    struct CollisionLog {
        CollType collflag = CollType::NOCOLLISION;
        size_t collind[2] = {0, 0}; // Indices of particles
        double colltime = -1.0;
        double colltime_ds = -1.0;

        //Default constructor initializes all elements to null
        CollisionLog() {
            collflag = CollType::NOCOLLISION;
            collind[0] = collind[1] = 0;
            colltime = colltime_ds = -1.0;
        }

        //Default destructor
        ~CollisionLog() = default;

        void reset_collision() {
            this->collflag = CollType::NOCOLLISION;
            this->collind[0] = this->collind[1] = 0;
        }

        bool has_collided() const {
            return static_cast<bool>(this->collflag);
        }
    };

    struct PTDE_event {
        double rmin = 1e33;
        double semi = 0.0;
        double ecc = -1;
        size_t ibh = 0;
    };

    struct PTDE_history {
        size_t ind = 0;
        std::vector<PTDE_event> PTDE_list;
        PTDE_event last_PTDE;
        bool ptde_candidate = false;
    };

    struct PTDELog {
        std::vector<size_t> stars_ind;
        std::map<size_t, PTDE_history> PTDEHistories;
        bool ptde_candidate = false;
        bool had_ptde = false;
    };

}

#endif //TSUNAMI_NBODYUTILS_H
