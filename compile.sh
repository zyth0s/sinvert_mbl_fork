
# Compile PETSC
mkdir petsc
cd petsc
git clone -b release htts://gitlab.com/petsc/petsc.git
export PETSC_DIR=$PWD
export PETSC_ARCH=linux-real-mumps
./configure --download-scalapack --download-metis --download-mumps --download-mpich
make all

cd ..

# Compile SLEPC
mkdir slepc
cd slepc
git clone -b release htts://gitlab.com/slepc/slepc.git
export SLEPC_DIR=$PWD
./configure
make all

cd ..

# Compile sinvert_mbl
mkdir build
cd build
cmake -DMACHINE=linux ../src
make

