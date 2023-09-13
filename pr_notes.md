## Ian's mutations

- module to get surface searchlight indices
- module to get volume searchlight indices
- module to compute brain rdms from searchlight 
indices and betas  
- re-pack dissimilarities in rdm object list to accomodate eval
- handle native, or fsaverage data preparation spaces.
- refactor get_distance so that it flexibly uses the rsatoolboc compute_rdms functionalities
- add nilearn to requirements

## jasper's todo's

- [x] test masking on sitek script
- [x] test masking on jones script
- [x] run daniel's tutorial with current code to see if that fails too??
- [x] remove deprecated fns demo
- [ ] bold5000 for surface?
- [ ] remove chunking "else" case of fewer than 1000 centres
- [ ] evaluate fail silently if nan's
- [ ] subpress divide by zero warning in calc_corr
- [ ] remove these notes
- [ ] remove ad hoc test scripts
- [ ] mp vs jl



## discussion

- searchlight subpackage, .pipeline, some in .io
- catch nans early: yes (in searchlight)
- reason not to split into stages: potential to parallelize across nodes, need to not load verything in memory

```
# ## this is the part that throws the error about nans.
# ## the evaluate fn calls compare() which checks that both rdms only have nans
# ## in the same locations. 
# rdm1 = tone_model.predict_rdm(theta=None)
# rdm2 = SL_RDM[1_000_000]
# vector1 = rdm1.get_vectors()
# vector2 = rdm2.get_vectors()
# from rsatoolbox.rdm.compare import _parse_input_rdms
# _parse_input_rdms(rdm1, rdm2)
#eval_fixed(tone_model, SL_RDM[1_000_000], method='spearman', theta=None)
```

