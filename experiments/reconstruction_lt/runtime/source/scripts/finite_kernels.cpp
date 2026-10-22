// Exact integer finite-stream kernels. No expectation or noise surrogate.
#include <cstdint>
#include <vector>
#include <algorithm>

extern "C" {
void fsm_update(int16_t *state, const uint8_t *bits, int16_t *trace,
                int64_t count, int64_t width, int32_t states) {
    for (int64_t t=0;t<width;t++) {
        for (int64_t i=0;i<count;i++) {
            int s=state[i]+(bits[t*count+i] ? 1 : -1);
            state[i]=std::min(std::max(s,0),states-1);
            if (trace) trace[t*count+i]=state[i];
        }
    }
}

void tt_events(const int16_t *trace, const double *uniforms,
               const double *cdfs, const int32_t *right_sizes,
               int32_t *counts, int64_t rows, int64_t width,
               int32_t states, int32_t n, int32_t pad) {
    int64_t events=rows*width;
    for (int64_t e=0;e<events;e++) {
        int bond=0;
        for (int j=0;j<n;j++) {
            const double *cdf=cdfs+((int64_t(j)*states+trace[e*n+j])*pad+bond)*pad;
            double u=uniforms[int64_t(j)*events+e];
            int lo=0,hi=right_sizes[j];
            while (lo<hi) {
                int mid=(lo+hi)/2;
                if (cdf[mid]<u) lo=mid+1; else hi=mid;
            }
            bond=std::min(lo,right_sizes[j]-1);
        }
        counts[(e%rows)*n+bond]++;
    }
}

void scalar_gates(int16_t *state, const uint8_t *drive,
                  const uint32_t *mag_shift, const uint32_t *em_shift,
                  const uint32_t *mag_net, const uint32_t *em_net,
                  const uint64_t *mag_threshold, const uint64_t *table_threshold,
                  const int32_t *map, const int32_t *sign,
                  int32_t *counts, int64_t nx, int64_t np,
                  int64_t width, int32_t states) {
    std::vector<uint8_t> gate(np);
    for (int64_t t=0;t<width;t++) {
        for (int64_t p=0;p<np;p++) {
            int s=state[p]+(drive[t*np+p] ? 1 : -1);
            state[p]=std::min(std::max(s,0),states-1);
            gate[p]=uint64_t(em_shift[p]^em_net[t])<table_threshold[state[p]];
        }
        for (int64_t i=0;i<nx;i++) {
            if (gate[map[i]] && uint64_t(mag_shift[i]^mag_net[t])<mag_threshold[i])
                counts[i]+=sign[i];
        }
    }
}

void bit_dot(const uint64_t *__restrict__ x, const uint64_t *__restrict__ w,
             const int32_t *__restrict__ xs, const int32_t *__restrict__ ws,
             int32_t *__restrict__ result, int64_t batches, int64_t rows,
             int64_t outputs, int64_t features, int64_t words) {
    for (int64_t b=0;b<batches;b++) {
        const uint64_t *xb=x+b*rows*features*words;
        const uint64_t *wb=w+b*outputs*features*words;
        const int32_t *sx=xs+b*rows*features;
        const int32_t *sw=ws+b*outputs*features;
        for (int64_t i=0;i<rows;i++) {
            for (int64_t j=0;j<outputs;j++) {
                int32_t total=0;
                for (int64_t k=0;k<features;k++) {
                    int s=sx[i*features+k]*sw[j*features+k];
                    if (s==0) continue;
                    const uint64_t *a=xb+(i*features+k)*words;
                    const uint64_t *c=wb+(j*features+k)*words;
                    int32_t p=0;
                    for (int64_t l=0;l<words;l++) p+=__builtin_popcountll(a[l]&c[l]);
                    total+=s*p;
                }
                result[(b*rows+i)*outputs+j]=total;
            }
        }
    }
}
}
