---
title: 'PyEEMA: A Python library for the preprocessing, correction, deconvolution and analysis of Excitation Emission Matrices (EEMs).'
tags:
 - python
 - fluorescence
 - spectroscopy
 - excitation emission matrix
 - environmental monitoring
authors:
 - name: Drew Meyers
   affiliation: 1
 - name: Qinmin Zheng
   affiliation: 2
 - name: Fabio Duarte
   affiliation: "2, 3"
 - name: Harold H Hemond
   affiliation: 1
 - name: Carlo Ratti
   affiliation: 2  
 - name: Andrew J Whittle
   affiliation: 1
affiliations:
 - name: Department of Civil and Environmental Engineering, Massachusetts Institute of Technology
   index: 1
 - name: Senseable City Lab, Massachusetts Institute of Technology
   index: 2
 - name: Pontifícia Universidade Católica do Paraná, Brazil
   index: 3
date: 2020-07-08
bibliography: paper.bib
---

# Statement of Need

A clear Statement of Need that illustrates the research purpose of the software...

Fluorescence Excitation and Emission Matrix Spectroscopy (EEMs) is a popular analytical technique in environmental monitoring. In particular, it has been applied extensively to investigate the composition and concentration of dissolved organic matter (DOM) in aquatic systems [sourcse]. Historically, EEMs have been combined with multi-way techniques such as PCA, ICA, and PARAFAC in order to decompose chemical mixtures [sources]. More recently, machine learning approaches such as convolutional neural networks (CNNs) and autoencoders have been applied to EEMs for source sepearation of chemical mixtures [sources]. However, before these source separation techniques can be performed, several preprocessing and correction steps must be applied to the raw EEMs [sources]. In order to achieve comparability between studies, standard methods to apply these corrections have been developed [sources]. These standard methods have been implemented in Matlab and R packages [sources]. However until PyEEMA, no Python package existed which implemented these standard correction steps. Furthermore, the Matlab and R implementations impose metadata schemas on users which limit their ability to track several important metrics corresponding with each measurement set. By providing a Python implementation, researchers will now be able to more effectively leverage Python's large scienfitic computing ecosystem when working with EEMs. In addition to the implementation of the preprocessing and correction steps, PyEEMA also provides researchers with the ability to create augmented mixture and single source training data from a small set of calibration EEM measurements. The augmentation technique relies on the fact that fluorescnce spectra are linearly additive, according to Beer's law [source]. This augmentation technique was first described in Rutherford et al., in which it was used to train a CNN to predict the concentration of single sources of pollutants in spectral mixtures [source]. Additionally, augmented and synthetic data has shown promise in improving the performace of deep learning models in several fields [sources]. PyEEMA presents the first open source implementation of such an augmentation technique for EEMs. PyEEMA also provides visualization toolbox useful in the interpretation of EEMs.

# Summary

A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience...
Enables some new research challenges to be addressed or makes addressing research challenges significantly better (e.g., faster, easier, simpler)...
Feature-complete (i.e. no half-baked solutions) and designed for maintainable extension (not one-off modifications of existing tools)...

PyEEMA is a python library for the preprocessing, correction, deconvolution and analysis of Excitation Emission Matrices (EEMs)...

Supported instruments, example datasets
Metadata schema
Cropping, blank subtraction, scattering removal, inner-filter effect correction, Raman normalization
Augmentation
Visualization

# Acknowledgements

We acknowledge contributions from...

# References

A list of key references, including to other software addressing related needs...
