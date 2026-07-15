"""Dataset loaders for the two-stage training pipeline.

RoboCloth capture data:
    MultiMaterialDenseDataset - stage 1 (dense per-point observation tensors)
    RealImageDenseDataset     - stage 2 training views
    RealValDataset            - stage 2 / evaluation held-out views
Comparison datasets:
    MERLBRDFIterableDataset / MERLBRDFFixedDataset - MERL measured BRDFs
    BonnDataset / BonnValDataset / BonnSingleMaterial* - Bonn UBOFAB19 SVBRDFs
    UBOBTFTrainDataset / UBOBTFValDataset - UBO2014 BTFs
"""
from .real import RealValDataset, RealImageDenseDataset
from .points import MultiMaterialDenseDataset
from .merl import MERLBRDFIterableDataset, MERLBRDFFixedDataset
from .bonn import BonnDataset, BonnValDataset, BonnSingleMaterialDataset, BonnSingleMaterialValDataset
from .ubo import UBOBTFTrainDataset, UBOBTFValDataset
