rm -rf build
mkdir build
/opt/rocm/bin/amdclang++ -x assembler -target amdgcn-amd-amdhsa -mcode-object-version=4 -mcpu=gfx950 -mwavefrontsize64 -c -g -o build/matmul.o matmul.s
/opt/rocm/bin/amdclang++ -target amdcgn-amdhsa build/matmul.o -o build/matmul.co
hipcc main.cpp -o build/matmul
