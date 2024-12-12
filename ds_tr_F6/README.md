```bash
for ds_read_b96_tr_b6, need to pad 32bits per 96bits to accomplish 128bits alignment requirement
normal ds_read_b96 doesn't need alignment requirement

if cmake available:
  modify matmul.s abd asm.cmake gfx version
  cd [exmaple]
  mkdir build
  cd build
  cmake .. -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_C_COMPILER=hipcc -DCMAKE_PREFIX_PATH=/opt/rocm/lib/cmake
  make -j
else :
  modify matmul.s gfx version
  ./build.sh [gfx version]
  ./build/matmul
```
