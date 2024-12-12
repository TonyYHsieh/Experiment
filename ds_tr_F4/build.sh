rm -rf build
mkdir build
/opt/rocm/bin/amdclang++ -x assembler -target amdgcn-amd-amdhsa -mcode-object-version=4 -mcpu=gfx950 -mwavefrontsize64 -c -g -o build/transpose.o transpose.s
/opt/rocm/bin/amdclang++ -target amdcgn-amdhsa build/transpose.o -o build/transpose.co
hipcc main.cpp -o build/transpose
