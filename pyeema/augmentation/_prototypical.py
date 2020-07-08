import numpy as np
import random
import itertools
import pandas as pd
import os

# TODO - make sure this should be singular...
# prototypical_spectrum()
def prototypical_spectra(source_name, source_df, cal_df, hdf):
    """Weighted average of calibration spectra with 
    randomly assigned weights between 0 and 1

    Returns:
        [type] -- [description]
    """

    proto_eems = []
    for index, row in source_df.iterrows():
        sample_set = str(index[0])
        sample_name = row["filename"]
        # TODO Read from 'cleaned' dir
        eem_path = os.path.join(*[
            "/corrections/sample_sets_raman_normalization",
            sample_set,
            sample_name
        ])
        '''
        eem_path = os.path.join(*[
            "corrections/sample_sets_discrete_excitations",
            sample_set,
            sample_name
        ])
        '''
        eem = pd.read_hdf(hdf, key=eem_path)
        proto_eems.append(eem)

    proto_concentration = cal_df["proto_conc"].values.item()
    if source_df[source_name].mean() != proto_concentration:
        raise Exception("Mismatch between prototypical concentration "
                        "given in calibration.csv and meta.csv")

    weights = []
    for i in range(len(proto_eems)):
        weights.append(random.uniform(0, 1))

    proto_eem = np.average([eem.values for eem in proto_eems],
                           axis=0, weights=weights)

    proto_eem = pd.DataFrame(data=proto_eem,
                             index=proto_eems[0].index,
                             columns=proto_eems[0].columns)
    proto_eem.index.name = 'emission_wavelength'
    new_indices = np.array(['source',  'proto_conc'])
    proto_eem = proto_eem.assign(**{
        'source': source_name,
        'proto_conc': proto_concentration,
    })
    proto_eem.set_index(new_indices.tolist(), append=True, inplace=True)
    new_indices = np.append(new_indices, ('emission_wavelength'))
    proto_eem = proto_eem.reorder_levels(new_indices)
    proto_eem.to_hdf(hdf, key=os.path.join(*[
        "augmented", "prototypical_spectra", source_name
    ]))

    return proto_eem


def single_sources(source_name, sources, c, hdf,
                   conc_range, num_spectra):
    """[summary]

    Arguments:
        proto_eem {[type]} -- [description]
        cal_fit {[type]} -- [description]
        conc_range {[type]} -- [description]
        num_steps {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    proto_eem = pd.read_hdf(hdf, key=os.path.join(*[
        "augmented", "prototypical_spectra", source_name
    ]))

    proto_concentration = proto_eem.index.get_level_values(
        'proto_conc').unique().item()
    proto_eem.reset_index(level=['proto_conc'], drop=True, inplace=True)

    source_cal_coeffs = c.loc[:,
                              c.columns.str.startswith("cal_func_term")
                              ].iloc[0].values
    cal_func = np.poly1d(source_cal_coeffs)
    number_range = np.linspace(conc_range[0], conc_range[1], num=num_spectra)

    aug = []
    for new_concentration in number_range:
        scalar = cal_func(new_concentration) / cal_func(proto_concentration)
        ss_eem = proto_eem*scalar
        label = np.zeros(3)
        source_index = np.where(sources == source_name)
        label[source_index] = new_concentration

        ss_eem.index.name = 'emission_wavelength'
        ss_eem = ss_eem.assign(
            **dict(zip(sources, label))
        )

        new_indices = sources
        ss_eem.set_index(
            new_indices.tolist(), append=True, inplace=True)
        new_indices = np.insert(new_indices, 0, 'source', axis=0)
        new_indices = np.append(new_indices, ('emission_wavelength'))
        ss_eem = ss_eem.reorder_levels(new_indices)
        aug.append(ss_eem)

    aug_ss_df = pd.concat(aug)
    aug_ss_df.to_hdf(hdf, key=os.path.join(*[
        "augmented", "single_sources", source_name
    ]))

    return aug_ss_df


def mixtures(sources, cal_df, hdf, conc_range,
             num_steps, scale='linear'):
    """Rutherford et al. conc_range=(0.01, 6.3), num_steps=15,
    scale=logarithmic

    Arguments:
        sources {[type]} -- [description]
        sources_cal_coeffs {[type]} -- [description]
        sources_eems {[type]} -- [description]
        conc_range {[type]} -- [description]
        num_steps {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    proto_spectra = []
    for source in sources:
        proto_eem = pd.read_hdf(hdf, key=os.path.join(
            *["augmented", "prototypical_spectra", source]))
        proto_spectra.append(proto_eem)
    proto_eem_df = pd.concat(proto_spectra)

    if scale == "logarithmic":
        number_range = np.geomspace(
            conc_range[0], conc_range[1], num=num_steps)
    elif scale == "linear":
        number_range = np.linspace(
            conc_range[0], conc_range[1], num=num_steps)
    else:
        raise ValueError("scale must be 'logarithmic' or 'linear'")

    cartesian_product = [p for p in itertools.product(
        number_range.tolist(), repeat=3)]

    aug = []
    for conc_set in cartesian_product:
        mix = []
        for index, label in enumerate(zip(sources, conc_set)):
            source_name = label[0]
            new_concentration = label[1]

            c = cal_df[cal_df["source"] == source_name]
            source_cal_coeffs = c.loc[:,
                                      c.columns.str.startswith("cal_func_term")
                                      ].iloc[0].values
            cal_func = np.poly1d(source_cal_coeffs)

            proto_eem = proto_eem_df.xs(
                source_name, level='source', drop_level=False)

            proto_concentration = proto_eem.index.get_level_values(
                'proto_conc').unique().item()
            proto_eem.reset_index(
                level=['proto_conc'], drop=True, inplace=True)

            scalar = cal_func(new_concentration) / \
                cal_func(proto_concentration)
            new_eem = proto_eem*scalar
            mix.append(new_eem)

        mix_eem = pd.concat(mix).sum(level="emission_wavelength")
        mix_eem = mix_eem.assign(
            **dict(zip(sources, conc_set))
        )

        new_indices = sources
        mix_eem.set_index(
            new_indices.tolist(), append=True, inplace=True)
        new_indices = np.append(new_indices, ('emission_wavelength'))
        mix_eem = mix_eem.reorder_levels(new_indices)
        aug.append(mix_eem)

    aug_mix_df = pd.concat(aug)
    aug_mix_df.to_hdf(hdf, key=os.path.join(*[
        "augmented", "mixtures"
    ]))
    return aug_mix_df