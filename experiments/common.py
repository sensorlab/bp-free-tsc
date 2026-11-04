"""Dataset list, seeds and command-line options shared by all experiment runners."""

import argparse

DATASETS = [
    "ACSF1", "Adiac", "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "BME", "Car", "CBF", "Chinatown",
    "ChlorineConcentration", "CinCECGTorso", "Coffee", "Computers", "CricketX", "CricketY", "CricketZ", "Crop",
    "DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect", "DistalPhalanxTW",
    "Earthquakes", "ECG200", "ECG5000", "ECGFiveDays", "ElectricDevices", "EOGHorizontalSignal", "EOGVerticalSignal",
    "EthanolLevel", "FaceAll", "FaceFour", "FacesUCR", "FiftyWords", "Fish", "FordA", "FordB", "FreezerRegularTrain",
    "FreezerSmallTrain", "GunPoint", "GunPointAgeSpan", "GunPointMaleVersusFemale", "GunPointOldVersusYoung", "Ham",
    "HandOutlines", "Haptics", "Herring", "HouseTwenty", "InlineSkate", "InsectEPGRegularTrain", "InsectEPGSmallTrain",
    "InsectWingbeatSound", "ItalyPowerDemand", "LargeKitchenAppliances", "Lightning2", "Lightning7", "Mallat", "Meat",
    "MedicalImages", "MiddlePhalanxOutlineAgeGroup", "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW",
    "MixedShapesRegularTrain", "MixedShapesSmallTrain", "MoteStrain", "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2", "OliveOil", "OSULeaf", "PhalangesOutlinesCorrect", "Phoneme", "PigAirwayPressure",
    "PigArtPressure", "PigCVP", "Plane", "PowerCons", "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxTW", "RefrigerationDevices", "Rock", "ScreenType", "SemgHandGenderCh2", "SemgHandMovementCh2",
    "SemgHandSubjectCh2", "ShapeletSim", "ShapesAll", "SmallKitchenAppliances", "SmoothSubspace",
    "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "StarLightCurves", "Strawberry", "SwedishLeaf", "Symbols",
    "SyntheticControl", "ToeSegmentation1", "ToeSegmentation2", "Trace", "TwoLeadECG", "TwoPatterns", "UMD",
    "UWaveGestureLibraryAll", "UWaveGestureLibraryX", "UWaveGestureLibraryY", "UWaveGestureLibraryZ", "Wafer", "Wine",
    "WordSynonyms", "Worms", "WormsTwoClass", "Yoga"
]

# Seed 3 uses the official UCR train/test split, seeds 4-8 are stratified resplits
# with the official test-set size (see bpfree/data.py).
SEEDS = [3, 4, 5, 6, 7, 8]
EPOCHS = 600


def parse_args():
    parser = argparse.ArgumentParser(description="Run one experiment configuration over UCR datasets and seeds.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, help="UCR dataset names (default: all 112)")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="split seeds (default: 3 4 5 6 7 8)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="training epochs (default: 600)")
    args = parser.parse_args()
    unknown = sorted(set(args.datasets) - set(DATASETS))
    if unknown:
        parser.error(f"unknown datasets: {', '.join(unknown)}")
    return args
