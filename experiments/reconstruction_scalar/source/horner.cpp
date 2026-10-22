#include <cstdint>
static inline uint32_t next(uint32_t &s) {s ^= s<<13;s ^= s>>17;s ^= s<<5;return s;}
// Actual input/constant Bernoulli bits and NAND gates. No exp evaluation.
extern "C" void horner_counts(const uint64_t *x, const uint64_t *c,
 const uint32_t *initial, int32_t *out, int size, int L, int degree, int factors) {
 for(int i=0;i<size;i++) {
  uint32_t rng=initial[i];int count=0;
  for(int t=0;t<L;t++) {
   int product=1;
   for(int f=0;f<factors;f++) {
    int h=1;
    for(int k=degree;k>=1;k--) {
     int xb=(uint64_t)next(rng)<x[i];
     int cb=(uint64_t)next(rng)<c[k-1];
     h=1-(xb & cb & h);
    }
    product &= h;
   }
   count+=product;
  }
  out[i]=count;
 }
}
