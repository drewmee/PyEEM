---
title: 'PyEEM: A Python library for the preprocessing, correction, and analysis of Excitation Emission Matrices (EEMs).'
tags:
 - python
 - fluorescence
 - spectroscopy
 - excitation emission matrix
 - environmental monitoring
authors:
 - name: Drew Meyers
   affiliation: "1, 2"
 - name: Jay W Rutherford
   affiliation: 3
 - name: Qinmin Zheng
   affiliation: 2
 - name: Fabio Duarte
   affiliation: "2, 4"
 - name: Carlo Ratti
   affiliation: 2  
 - name: Harold H Hemond
   affiliation: 1
 - name: Andrew J Whittle
   affiliation: 1
affiliations:
 - name: Department of Civil and Environmental Engineering, Massachusetts Institute of Technology
   index: 1
 - name: Senseable City Lab, Massachusetts Institute of Technology
   index: 2
 - name: Department of Chemical Engineering, University of Washington
   index: 3
 - name: Pontifícia Universidade Católica do Paraná, Brazil
   index: 4
date: 2020-07-08
bibliography: paper.bib
---

# Summary

Fluorescence Excitation and Emission Matrix Spectroscopy (EEMs) is a popular analytical technique in environmental monitoring. In particular, it has been applied extensively to investigate the composition and concentration of dissolved organic matter (DOM) in aquatic systems [@Coble1990;@McKnight2001;@Fellman2010]. Historically, EEMs have been combined with multi-way techniques such as PCA, ICA, and PARAFAC in order to decompose chemical mixtures [@Bro1997;@Stedmon2008;@Murphy2013;@CostaPereira2018]. More recently, deep learning approaches such as convolutional neural networks (CNNs) and autoencoders have been applied to EEMs for source separation of chemical mixtures [@Cuss2016;@Peleato2018;@Ju2019;@Rutherford2020]. However, before these source separation techniques can be performed, several preprocessing and correction steps must be applied to the raw EEMs. In order to achieve comparability between studies, standard methods to apply these corrections have been developed [@Ohno2002;@Bahram2006;@Lawaetz2009;@R.Murphy2010;@Murphy2011;@Kothawala2013]. PyEEM provides a Python implementation for these standard preprocessing and correction steps for EEM measurements produced by several common spectrofluorometers.

In addition to the implementation of the standard preprocessing and correction steps, PyEEM also provides researchers with the ability to create augmented single source and mixture training data from a small set of calibration EEM measurements. The augmentation technique relies on the fact that fluorescence spectra are linearly additive in mixtures, according to Beer's law. This augmentation technique was first described in Rutherford et al., in which it was used to train a CNN to predict the concentration of single sources of pollutants in spectral mixtures [@Rutherford2020]. Additionally, augmented and synthetic data has shown promise in improving the performance of deep learning models in several fields [@Nikolenko2019]. 

Finally, PyEEM provides an extensive visualization toolbox, based on Matplotlib, which is useful in the interpretation of EEM datasets. This visualization toolbox includes various ways of plotting EEMs, the visualization of the Raman scatter peak area over time, and more.

# Statement of Need

Prior to PyEEM, no open source Python package existed to work with EEMs. However, such libraries have existed for MATLAB and R for some time [@Murphy2013;@Massicotte;Pucher2019]. By providing a Python implementation, researchers will now be able to more effectively leverage Python's large scientific computing ecosystem when working with EEMs. Furthermore, the existing libraries in MATLAB and R do not provide deep learning techniques for decomposing chemical mixtures from EEMs. These libraries provide PARAFAC methods for performing such a task. However, although this technique has been widely used for some time, it has its limitations and recent work has shown promise in using deep learning approaches. For this reason, PyEEM provides a toolbox for generating augmented training data as well as an implementation of the CNN architecture reported in Rutherford et al., which has shown to be able to successfully decompose spectral mixtures [@Rutherford2020].

# References