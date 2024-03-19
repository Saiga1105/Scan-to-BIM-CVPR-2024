# Scan-to-BIM-CVPR-2024
4th International Scan-to-BIM competition KUL and FBK repo with CODE.


![Alt text](/docs/assets/IMG_Stan_00_General.png "1")


We split the process into two steps, a **Detection** and a **Reconstruction** step. Each step is explained below. First, we start with the instance segmentation of the columns, walls and doors classes.


- Item 1 Semantic Segmentation: PTV3+PPT is an excellent baseline model for unstructured points clouds such as walls and column. However, it doesn't do instance segmentation and thus a clustering must be implemented to achieve an instance segmentation. 
    

- Item 2 Instance Segmentation: We can also try to directly compute instances. Good baseline models for intance segmentation OneFormer3D, Mask3D, and PBNet.

- Item 3 Object Detection: As Doors are unlikely to be found SAM is a proficient object detection model and is highly likely to find the doors. 

- Item 4 Filter the results: To improve the detection rate, we will impose constraints on the detected instances, specifically, the following conditions will be placed on the detected intances:
    - Subitem 4.1 Walls cannot the thicker than 0.5m
    - Subitem 4.2 Columns cannot have dimensions larger than w=0.5m and l=0.5m
    - Subitem 4.3 Doors cannot have dimensions larger than h=2.3m and w=1.2m




Second, we will perform the reconstruction step. Per convention, walls and columns will be modeled between reference levels, and as such, the first reconstructed ele


-  Item 1 Reference Levels: For the reference levels, we consider the architectural levels since we are modeling the visible construction elements in the architectural domain.

