#include <cstdint>
#include <algorithm>

// Literal stochastic gates/state updates, no Gaussian/noise/expectation surrogate.
static inline uint32_t step(uint32_t &s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
}
extern "C" void sexp_counts(const uint64_t *thresholds, const uint32_t *seeds,
    const uint8_t *exact_one, int32_t *counts, int64_t size,
    int length, int burn, int states, int gain) {
    for (int64_t j=0;j<size;++j) {
        if (exact_one[j]) { counts[j]=length; continue; }
        uint32_t rng=seeds[j]; int s=(states-1)/2, c=0;
        for (int t=0;t<burn+length;++t) {
            s += uint64_t(step(rng)) < thresholds[j] ? 1 : -1;
            s=std::min(states-1,std::max(0,s));
            if(t>=burn) c += s < states-gain;
        }
        counts[j]=c;
    }
}
extern "C" void bernstein_counts(const uint64_t *thresholds, const uint32_t *seeds,
    const uint64_t *coeff, int32_t *counts, int64_t size, int length, int degree) {
    for (int64_t j=0;j<size;++j) {
        uint32_t rng=seeds[j]; int c=0;
        for(int t=0;t<length;++t) {
            int k=0;
            for(int d=0;d<degree;++d) k += uint64_t(step(rng)) < thresholds[j];
            c += uint64_t(step(rng)) < coeff[k];
        }
        counts[j]=c;
    }
}
