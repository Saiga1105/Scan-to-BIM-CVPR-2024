# Scan-to-BIM-CVPR-2024
This is the KUL and FBK repo for the [4th International Scan-to-BIM competition](https://cv4aec.github.io/) .


![Alt text](/docs/assets/IMG_Stan_00_General.png "1")


We split the process into two steps, a **Detection** and a **Reconstruction** step. Each step is explained below. First, we start with the instance segmentation of the primary (walls, ceilings, floors, columns) and secondary structure classes (windows & doors). In total, 9 tasks are defined. 

## Detection

![Alt text](/docs/assets/detection.PNG "detection")

- T1. Semantic Segmentation: PTV3+PPT is an excellent baseline model for unstructured points clouds such as walls and column. However, it doesn't do instance segmentation and thus a clustering must be implemented to achieve an instance segmentation. 
    

- T2. Instance Segmentation: We can also try to directly compute instances. Good baseline models for intance segmentation OneFormer3D, Mask3D, and PBNet.

- T3. Object Detection: As Doors are unlikely to be found SAM is a proficient object detection model and is highly likely to find the doors. 

- T4. Filter the results: To improve the detection rate, we will impose constraints on the detected instances, specifically, the following conditions will be placed on the detected intances:
    - 4.1 Walls cannot the thicker than 0.5m
    - 4.2 Columns cannot have dimensions larger than w=0.5m and l=0.5m
    - 4.3 Doors cannot have dimensions larger than h=2.3m and w=1.2m


## Reconstruction
![Alt text](/docs/assets/reconstruction.PNG "reconstruction")

Second, we will perform the reconstruction step. Per convention, walls and columns will be modeled between reference levels, and as such, the first reconstructed ele


-  T5. Reference Levels: For the reference levels, we consider the architectural levels since we are modeling the visible construction elements in the architectural domain.

-  T6. [IfcWallStandardCase](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcsharedbldgelements/lexical/ifcwallstandardcase.htm): IfcBuildingElement with the following parameters. Only straight walls are reconstructed in this repo
    - IfcLocalPlacement (m): the two control points at both ends of the wall axis. Note that the wall axis is at the center of the wall. 
    - Wall Thickness (m): uniform distance between both wall faces.
    - base constraint (URI): bottom reference level
    - base offset (m): offset from the base constraint level to the bottom of the IfcBuildingElement
    - top constraint (URI): top reference level
    - top offset (m): offset from the base constraint level to the bottom of the IfcBuildingElement
    - HasOpenings (URI): links through [IfcRelVoidsElement](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcproductextension/lexical/ifcrelvoidselement.htm) to [IfcOpeningElement](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcproductextension/lexical/ifcopeningelement.htm) objects that define holes in the wall. 

-  T7. [IfcOpeningElement](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcproductextension/lexical/ifcopeningelement.htm): Child element of [IfcWallStandardCase](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcsharedbldgelements/lexical/ifcwallstandardcase.htm) that defines a boolean operation of subtraction between the geometric bodies of the element and the opening. IfcBuildingElement with the following parameters.
    - 

-  T7. [IfcColumn](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcsharedbldgelements/lexical/ifccolumn.htm):  IfcBuildingElement with the following parameters.
    - IfcCircleProfileDef.Radius (m): radius
    - IfcLocalPlacement (m): center of column at the base of the column
    - base constraint (URI): bottom reference level
    - base offset (m): offset from the base constraint level to the bottom of the IfcBuildingElement
    - top constraint (URI): top reference level
    - top offset (m): offset from the top constraint level to the top of the IfcBuildingElement

-  T8. Reference Levels: 