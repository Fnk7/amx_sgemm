#!/bin/bash

mkdir -p result

echo "Benchmarking with Accelerate"
mkdir -p build-accelerate
pushd build-accelerate
cmake -DDTYPE=FLOAT -DBACKEND=Accelerate -DCMAKE_BUILD_TYPE=Release ..
make
./gemm_bench > ../result/accelerate.dat
popd

echo "Benchmarking with OpenBLAS"
mkdir -p build-openblas
pushd build-openblas
LDFLAGS=-L$(brew --prefix openblas)/lib cmake -DDTYPE=FLOAT -DBACKEND=OpenBLAS -DCMAKE_BUILD_TYPE=Release ..
CPATH=$(brew --prefix openblas)/include make
./gemm_bench > ../result/openblas.dat
popd

for i in 1 2 3
do
	echo "Benchmarking with AMX $i"
	mkdir -p build-amx-$i
	pushd build-amx-$i
	cmake -DDTYPE=FLOAT -DBACKEND=AMX_$i -DCMAKE_BUILD_TYPE=Release ..
	make
	./gemm_bench > ../result/amx-$i.dat
	popd
done

echo "Benchmarking with Metal"
mkdir -p build-metal
pushd build-metal
cmake -DDTYPE=FLOAT -DBACKEND=Metal -DCMAKE_BUILD_TYPE=Release ..
make
./gemm_bench > ../result/metal.dat
popd

pushd result
gnuplot plot_gemm.gpi
