FALLACY-architecture
=======

*Fixing Amdahl\'s Law Within the Limits of Accelerated sYstems*

FALLACY is a memory centric toolset capable of detecting memory access pattern originating from multiple actors.
The *downward* facing tools provide architecture/hardware insight into the memory subsystems for various compute-memory organizations (CPU, GPU, FPGA, ASIC).
The *upward* facing tools provide pattern analysis based on statistical and machine learning models in support of analysis and attribution.

### Repository ###
*architecture*

we tend to development a memory centric toolset to map the memory trace to instruction level and detect the memory allocation under different memory subsystem.

### Version ###
0.1

### Authors (tags) ###
*Chenhao Xie ()*,
Andres Marquez (awmm)*,
Nathan Tallent ()*,
Ozgur Kilic ()*

### File Structure ###
```
fallacy-architecture/
|-- benchmarks
`-- trace
    |-- instruction 
    |-- memory
`-- tools
    |-- memory-sim
    |-- arch_models
    |-- mapping

```

# cuSparse
