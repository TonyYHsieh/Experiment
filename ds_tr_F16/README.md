```bash
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
