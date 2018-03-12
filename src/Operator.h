#ifndef OPERATOR_H
#define OPERATOR_H

#include "Basis.h"

class Operator
{
    private:
        Basis * baseptr;

    public:
        Operator(Basis * baseptr__):
            baseptr(baseptr__){}

        void apply_Siz(size_t site_idx, Vec * parallel_vec_ptr__);
};



void Operator::apply_Siz(size_t site_idx, Vec * parallel_vec_ptr__)
{
    PetscScalar * wavefunction_ptr;
    VecGetArray( *parallel_vec_ptr__, &wavefunction_ptr  );

    std::pair<PetscInt, PetscInt> mpirange;
    VecGetOwnershipRange(*parallel_vec_ptr__, &mpirange.first, &mpirange.second);

    int cntr = 0;
    for(size_t i=mpirange.first; i<mpirange.second; i++)
    {

        State state=baseptr->get_state(i);

        if(state[site_idx])
            wavefunction_ptr[cntr]*=  0.5;
        else
            wavefunction_ptr[cntr]*= -0.5;
        cntr++;
    }

    VecRestoreArray( *parallel_vec_ptr__, &wavefunction_ptr );
}



#endif
