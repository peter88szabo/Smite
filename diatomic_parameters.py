#Data from NIST chem:

diatomic_molecules = {
    "H2": {
    "req": 0.7414,  # in Å
    "omega": 4401.21,  # in cm⁻¹
    "De": 4.478  # in eV
    },

    "H2+": {
    "req": 1.06,  # in Å
    "omega": 2322.7,  # in cm⁻¹
    "De": 2.79  # in eV
    },

    # Hydrogen Halides (H-X)
    "HF": {  # X ^1Σ+
        "req": 0.917,  # in Å
        "omega": 3961.41,  # in cm⁻¹
        "De": 5.87  # in eV
    },
    "HCl": {  # X ^1Σ+
        "req": 1.27455,  # in Å
        "omega": 2990.946,  # in cm⁻¹
        "De": 4.43  # in eV
    },
    "HBr": {  # X ^1Σ+
        "req": 1.414,  # in Å
        "omega": 2649.7,  # in cm⁻¹
        "De": 3.79  # in eV
    },
    "HI": {  # X ^1Σ+
        "req": 1.609,  # in Å
        "omega": 2308.8,  # in cm⁻¹
        "De": 3.06  # in eV
    },

    # Hydroxyl Radical and Cation
    "OH": {  # X ^2Π
        "req": 0.969,  # in Å
        "omega": 3737.76,  # in cm⁻¹
        "De": 4.39  # in eV
    },
    "OH+": {  # X ^3Σ-
        "req": 1.033,  # in Å
        "omega": 2982,  # in cm⁻¹
        "De": 5.06  # in eV
    },

    # Oxides (O-X)
    "O2": {  # X ^3Σg-
        "req": 1.2075,  # in Å
        "omega": 1580.19,  # in cm⁻¹
        "De": 5.12  # in eV
    },
    "O2+": {  # X ^2Πg
        "req": 1.116,  # in Å
        "omega": 1904,  # in cm⁻¹
        "De": 6.36  # in eV
    },
    "CO": {  # X ^1Σ+
        "req": 1.128,  # in Å
        "omega": 2169.814,  # in cm⁻¹
        "De": 11.09  # in eV
    },
    "CO+": {  # X ^2Σ+
        "req": 1.115,  # in Å
        "omega": 2184.1,  # in cm⁻¹
        "De": 10.8  # in eV
    },
    "NO": {  # X ^2Π
        "req": 1.1508,  # in Å
        "omega": 1904.17,  # in cm⁻¹
        "De": 6.47  # in eV
    },
    "NO+": {  # X ^1Σ+
        "req": 1.063,  # in Å
        "omega": 2379.5,  # in cm⁻¹
        "De": 9.76  # in eV
    },
    "SO": {  # X ^3Σ-
        "req": 1.481,  # in Å
        "omega": 1150,  # in cm⁻¹
        "De": 5.45  # in eV
    },
    "SO+": {  # X ^2Π
        "req": 1.453,  # in Å
        "omega": 1201,  # in cm⁻¹
        "De": 5.96  # in eV
    },
    "SiO": {  # X ^1Σ+
        "req": 1.509,  # in Å
        "omega": 1241.9,  # in cm⁻¹
        "De": 8.26  # in eV
    },
    "PO": {  # X ^2Π
        "req": 1.476,  # in Å
        "omega": 1204,  # in cm⁻¹
        "De": 6.4  # in eV
    },
    "ClO": {  # X ^2Π
        "req": 1.57,  # in Å
        "omega": 854.9,  # in cm⁻¹
        "De": 2.52  # in eV
    },
    "BrO": {  # X ^2Π
        "req": 1.720,  # in Å
        "omega": 720,  # in cm⁻¹
        "De": 2.29  # in eV
    },
    "IO": {  # X ^2Π
        "req": 1.916,  # in Å
        "omega": 561.1,  # in cm⁻¹
        "De": 2.12  # in eV
    },
    "LiO": {  # X ^2Π
        "req": 1.606,  # in Å
        "omega": 924.2,  # in cm⁻¹
        "De": 2.48  # in eV
    },
    "NaO": {  # X ^2Π
        "req": 1.887,  # in Å
        "omega": 659,  # in cm⁻¹
        "De": 2.05  # in eV
    },
    "KO": {  # X ^2Π
        "req": 2.266,  # in Å
        "omega": 423.3,  # in cm⁻¹
        "De": 1.68  # in eV
    },
    "BeO": {  # X ^1Σ+
        "req": 1.331,  # in Å
        "omega": 1476,  # in cm⁻¹
        "De": 6.74  # in eV
    },
    "MgO": {  # X ^1Σ+
        "req": 1.749,  # in Å
        "omega": 787.4,  # in cm⁻¹
        "De": 4.87  # in eV
    },
    "CaO": {  # X ^1Σ+
        "req": 1.978,  # in Å
        "omega": 728.1,  # in cm⁻¹
        "De": 4.18  # in eV
    },
    "ZnO": {  # X ^1Σ+
        "req": 1.704,  # in Å
        "omega": 788.6,  # in cm⁻¹
        "De": 4.09  # in eV
    },
    "FeO": {  # X ^5Δ
        "req": 1.616,  # in Å
        "omega": 887,  # in cm⁻¹
        "De": 4.17  # in eV
    },
    "FeO+": {  # X ^6Σ+
        "req": 1.616,  # in Å
        "omega": 890,  # in cm⁻¹
        "De": 5.0  # in eV
    },
    "CuO": {  # X ^2Π
        "req": 1.725,  # in Å
        "omega": 725,  # in cm⁻¹
        "De": 4.01  # in eV
    },
    "CuO+": {  # X ^1Σ+
        "req": 1.75,  # Approximation, in Å
        "omega": 720,  # Approximation, in cm⁻¹
        "De": 4.8  # Approximation, in eV
    },
    "NiO": {  # X ^3Σ-
        "req": 1.627,  # in Å
        "omega": 870,  # in cm⁻¹
        "De": 4.4  # in eV
    },
    "NiO+": {  # X ^2Σ+
        "req": 1.625,  # Approximation, in Å
        "omega": 865,  # Approximation, in cm⁻¹
        "De": 5.1  # Approximation, in eV
    },

   #Sulfids:
    "HS": {  # X ^2Π
        "req": 1.34,  # in Å
        "omega": 2728.8,  # in cm⁻¹
        "De": 3.57  # in eV
    },
    "HS+": {  # X ^3Σ-
        "req": 1.383,  # in Å
        "omega": 2566,  # in cm⁻¹
        "De": 3.78  # in eV
    },
    "S2": {  # X ^3Σg-
        "req": 1.889,  # in Å
        "omega": 725.68,  # in cm⁻¹
        "De": 4.38  # in eV
    },
    "S2+": {  # X ^2Πg
        "req": 1.88,  # in Å
        "omega": 700,  # in cm⁻¹
        "De": 4.65  # in eV
    },
    "SiS": {  # X ^1Σ+
        "req": 2.015,  # in Å
        "omega": 758.9,  # in cm⁻¹
        "De": 5.12  # in eV
    },
    "SiS+": {  # X ^2Π
        "req": 1.986,  # in Å
        "omega": 769,  # in cm⁻¹
        "De": 5.30  # in eV
    },
    "PS": {  # X ^2Π
        "req": 1.904,  # in Å
        "omega": 731.8,  # in cm⁻¹
        "De": 5.06  # in eV
    },
    "PS+": {  # X ^1Σ+
        "req": 1.874,  # in Å
        "omega": 740,  # in cm⁻¹
        "De": 5.24  # in eV
    },
    "ClS": {  # X ^2Π
        "req": 2.018,  # in Å
        "omega": 455.6,  # in cm⁻¹
        "De": 3.52  # in eV
    },
    "ClS+": {  # X ^3Σ-
        "req": 2.001,  # in Å
        "omega": 462,  # in cm⁻¹
        "De": 3.70  # in eV
    },
    "BrS": {  # X ^2Π
        "req": 2.189,  # in Å
        "omega": 391,  # in cm⁻¹
        "De": 3.12  # in eV
    },
    "BrS+": {  # X ^3Σ-
        "req": 2.170,  # in Å
        "omega": 395,  # in cm⁻¹
        "De": 3.30  # in eV
    },
    "IS": {  # X ^2Π
        "req": 2.399,  # in Å
        "omega": 296.8,  # in cm⁻¹
        "De": 2.79  # in eV
    },
    "IS+": {  # X ^3Σ-
        "req": 2.380,  # in Å
        "omega": 300,  # in cm⁻¹
        "De": 2.95  # in eV
    },

    "HeH+": {
    "req": 0.772,  # in Å
    "omega": 2325,  # in cm⁻¹
    "De": 2.75  # in eV
    },


    "He2+": {  # X ^2Σu+
        "req": 1.08,  # in Å
        "omega": 2030,  # in cm⁻¹
        "De": 2.5  # in eV
    },
    "Ne2+": {  # X ^2Σu+
        "req": 1.12,  # in Å
        "omega": 1100,  # in cm⁻¹
        "De": 1.2  # in eV
    },
    "Ar2+": {  # X ^2Σu+
        "req": 1.20,  # in Å
        "omega": 630,  # in cm⁻¹
        "De": 1.5  # in eV
    },
    "Kr2+": {  # X ^2Σu+
        "req": 1.22,  # in Å
        "omega": 400,  # in cm⁻¹
        "De": 1.0  # in eV
    },
    "Xe2+": {  # X ^2Σu+
        "req": 1.30,  # in Å
        "omega": 250,  # in cm⁻¹
        "De": 0.8  # in eV
    },
    "HeNe+": {  # X ^2Σ+
        "req": 1.1,  # in Å
        "omega": 180,  # in cm⁻¹
        "De": 0.3  # in eV
    },
    "NeAr+": {  # X ^2Σ+
        "req": 1.18,  # in Å
        "omega": 280,  # in cm⁻¹
        "De": 0.4  # in eV
    },
    "ArKr+": {  # X ^2Σ+
        "req": 1.20,  # in Å
        "omega": 180,  # in cm⁻¹
        "De": 0.3  # in eV
    },
    "KrXe+": {  # X ^2Σ+
        "req": 1.30,  # in Å
        "omega": 150,  # in cm⁻¹
        "De": 0.25  # in eV
    }
}  



if __name__ == "__main__":
    data = diatomic_molecules["HCl"]

    print("HCl Molecular Data:")
    print(f"Equilibrium Bond Length (req): {data['req']} Å")
    print(f"Vibrational Frequency (omega): {data['omega']} cm-1")
    print(f"Dissociation Energy (De): {data['De']} eV\n")


    for molecule, data in diatomic_molecules.items():
        print(f"{molecule} Molecular Data:")
        print(f"  Equilibrium Bond Length (req): {data['req']} Å")
        print(f"  Vibrational Frequency (omega): {data['omega']} cm⁻¹")
        print(f"  Dissociation Energy (De): {data['De']} eV\n")
