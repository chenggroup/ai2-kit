#!/bin/bash
set -e

EVN_NAME=ec-MLP_devel

module purge

# download deepmd-kit v3.1.0a0
if [ ! -d ./deepmd-kit ]; then
    git clone -b devel https://gitee.com/deepmodeling/deepmd-kit.git
fi
export deepmd_source_dir=$(pwd)/deepmd-kit

module load intel/2023.2
module load anaconda/2022.5

# create conda environment if not exist
conda env list | grep -q $EVN_NAME || conda create -y -n $EVN_NAME python=3.11
source activate $EVN_NAME

export _LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export _PATH=$PATH
export _CONDA_PREFIX=$CONDA_PREFIX

module load cuda/12.1
module load cudnn/8.4.1
module load gcc/12.1

export CC=$(which gcc)
export CXX=$(which g++)
export FC=$(which gfortran)
export MPI_CXX=$(which mpicxx)
export MPI_CC=$(which mpicc)
export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"

export DP_VARIANT=cuda
export DP_ENABLE_PYTORCH=1
export CONDA_PREFIX=$_CONDA_PREFIX

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH=$CONDA_PREFIX/bin:$PATH

pip install tensorflow==2.18
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ninja cmake
pip install nvidia-cuda-nvcc-cu11

# install deepmd-kit python interface
[ -f deepmd-py.done ] || {
    pushd $deepmd_source_dir
    git checkout 99c9dc23
    wget -c https://patch-diff.githubusercontent.com/raw/ChiahsinChu/deepmd-kit/pull/1.patch
    git am 1.patch 
    git clean -fdx
    CMAKE_ARGS="$CMAKE_ARGS -DUSE_CUDA_TOOLKIT=TRUE" pip install .
    popd
    touch deepmd-py.done
}

# install deepmd-kit lammps plugin
module load gsl/2.8

# download if libtorch does not exist
if [ ! -d ./libtorch ]; then
    wget -c https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.1%2Bcu121.zip
    unzip -q libtorch-cxx11-abi-shared-with-deps-2.1.1+cu121.zip
    rm libtorch-*.zip
fi
export torch_root=$(pwd)/libtorch

# download if lammps-* directory does not exist
if [ ! -d ./lammps ]; then
    git clone -b electrode https://github.com/robeme/lammps.git
    
fi
export LAMMPS_PREFIX=$(pwd)/lammps

[ -f lammps.done ] || {
    pushd $LAMMPS_PREFIX
    mkdir -p build && cd build
    cmake ../cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_Fortran_COMPILER=gfortran \
        -D BUILD_MPI=yes -D BUILD_OMP=yes -D LAMMPS_MACHINE=mpi \
        -C ../cmake/presets/most.cmake -C ../cmake/presets/nolib.cmake \
        -D PKG_PLUGIN=yes -D PKG_MOLECULE=yes -D PKG_RIGID=yes \
        -D PKG_ELECTRODE=yes \
        -D PKG_KSPACE=yes -D PKG_EXTRA-DUMP=yes \
        -D BUILD_SHARED_LIBS=yes
    make -j4
    popd
    touch lammps.done
}


# if don't need other package
# module load lammps to avoid build lammps
export deepmd_root=$deepmd_source_dir/build/deepmd_root
export LAMMPS_PLUGIN_PATH=$deepmd_root/lib/deepmd_lmp
export LD_LIBRARY_PATH=$deepmd_root/lib:$LD_LIBRARY_PATH
export PATH=$LAMMPS_PREFIX/build:$PATH

[ -f deepmd-c.done ] || {
    pushd $deepmd_source_dir/source
    mkdir -p build && cd build
    cmake -DUSE_CUDA_TOOLKIT=ON \
        -DENABLE_TENSORFLOW=TRUE \
        -DUSE_TF_PYTHON_LIBS=TRUE \
        -DENABLE_PYTORCH=TRUE \
        -DCMAKE_PREFIX_PATH=$torch_root \
        -DCMAKE_INSTALL_PREFIX=$deepmd_root \
        -DCAFFE2_USE_CUDNN=TRUE \
        -DCUDNN_ROOT=/public/software/cudnn/8.4.1/ \
        -DLAMMPS_SOURCE_ROOT=$LAMMPS_PREFIX ..
    make -j4
    make install
    popd
    touch deepmd-c.done
}


if [ ! -d ./ec-MLP ]; then                                                              
   #git clone -b devel https://git.xmu.edu.cn/cheng-group/ec-MLP.git
   git clone -b devel git@git.xmu.edu.cn:cheng-group/ec-MLP.git
fi

if [ ! -d ./torch-admp ]; then                                                              
   git clone -b devel https://github.com/ChiahsinChu/torch-admp
fi

[ -f torch-admp.done ] || {
    pip install torch-admp
    touch torch-admp.done
}

export ec_MLP_source_dir=$(pwd)/ec-MLP                                             
[ -f ec-MLP.done ] || {
    pushd $ec_MLP_source_dir
    pip install .
    popd
    pushd $ec_MLP_source_dir/src/lmp
    mkdir -p build && cd build
    cmake -DLAMMPS_SOURCE_DIR=$LAMMPS_PREFIX/src \
      -DDEEPMD_SOURCE_DIR=$deepmd_source_dir/source/lmp \
      -DCMAKE_PREFIX_PATH=$deepmd_root \
      ..
    make -j4
    popd
    touch ec-MLP.done
}


mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

cat <<__EOF >$CONDA_PREFIX/etc/conda/activate.d/libdeepmd.sh
export _LD_LIBRARY_PATH=\$LD_LIBRARY_PATH
export _PATH=\$PATH
export _CONDA_PREFIX=\$CONDA_PREFIX
module load cuda/12.1 
module load cudnn/8.4.1
module load gcc/12.1
export CC=\$(which gcc)
export CXX=\$(which g++)
export FC=\$(which gfortran)
export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"
export DP_VARIANT=cuda
export DP_ENABLE_PYTORCH=1
export CONDA_PREFIX=\$_CONDA_PREFIX
unset _CONDA_PREFIX
export deepmd_source_dir=$deepmd_source_dir
export deepmd_root=\$deepmd_source_dir/build/deepmd_root
export LAMMPS_PLUGIN_PATH=\$deepmd_root/lib/deepmd_lmp
export LAMMPS_PLUGIN_PATH=\$HOME/software/ec-mlp/ec-MLP/src/lmp/build:$LAMMPS_PLUGIN_PATH
export LAMMPS_PREFIX=$LAMMPS_PREFIX
export torch_root=$torch_root
export LD_LIBRARY_PATH=\$deepmd_root/lib:\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH
export PATH=\$LAMMPS_PREFIX/build:\$CONDA_PREFIX/bin:\$PATH
#cat <<EOF 
#If you are to run LAMMPS,
#then you must add the following modules to your script.
#module load intel/2023.2
#module load gsl/2.8
#EOF
__EOF

cat <<__EOF >$CONDA_PREFIX/etc/conda/deactivate.d/libdeepmd.sh
module purge
unset DP_VARIANT
unset DP_ENABLE_PYTORCH
export LD_LIBRARY_PATH=\$_LD_LIBRARY_PATH
unset _LD_LIBRARY_PATH
export PATH=\$_PATH
unset _PATH
__EOF
