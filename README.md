# Scan-to-BIM-CVPR-2024
This is the KUL and FBK repo for the [4th International Scan-to-BIM competition](https://cv4aec.github.io/) .


![Alt text](/docs/assets/IMG_Stan_00_General.png "1")


We split the process into two steps, a **Detection** and a **Reconstruction** step. Each step is explained below. First, we start with the instance segmentation of the primary (walls, ceilings, floors, columns) and secondary structure classes (windows & doors). In total, 9 tasks are defined. 

## Detection

![Alt text](/docs/assets/detection.PNG "detection")

- **T1. Semantic Segmentation**: [PTV3+PPT](https://github.com/Pointcept/PointTransformerV3) is an excellent baseline model for unstructured points clouds such as walls and column. However, it doesn't do instance segmentation and thus a clustering must be implemented to achieve an instance segmentation. 
    

- **T2. Instance Segmentation**: We can also try to directly compute instances. Good baseline models for intance segmentation [OneFormer3D](https://github.com/oneformer3d/oneformer3d), [Mask3D](https://github.com/JonasSchult/Mask3D), and [PBNet](https://github.com/weiguangzhao/PBNet).

- **T3. Object Detection**: As doors are unlikely to be found in a point cloud, we will target them using image object detection with [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO).

- **T4. Filter the results**: To improve the detection rate, we will impose some constraints on the detected instances, specifically, the following conditions will be placed on the detected intances:
    - Walls cannot the thicker than 0.5m
    - Columns cannot have dimensions larger than w=0.5m and l=0.5m
    - Doors cannot have dimensions larger than h=2.3m and w=1.2m
    - [IfcBuildingStory](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcproductextension/lexical/ifcbuildingstorey.htm) cannot be closer than 3m apart and further apart than 6m
    - [IfcColumn](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcsharedbldgelements/lexical/ifccolumn.htm) max radius of 1m




## Reconstruction
![Alt text](/docs/assets/reconstruction.PNG "reconstruction")

Second, we will perform the reconstruction step. Per convention, walls and columns will be modeled between reference levels, and as such, the first reconstructed ele


-  **T5. [IfcBuildingStory](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcproductextension/lexical/ifcbuildingstorey.htm)**: For the reference levels, we consider the architectural levels (Finish Floor Level or FFL) since we are modeling the visible construction elements in the architectural domain.
    - IfcLocalPlacement (m): center point of the IfcBuildingElement
    - FootPrint (m): 2D lineset or parametric 2D orientedBoundingBox (c_x,c_y,c_z,R_z,s_u,s_v,s_w)
    - Elevation (m): c_z
    - resource (Open3D.TriangleMesh): plane of the storey

-  **T6. [IfcWallStandardCase](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcsharedbldgelements/lexical/ifcwallstandardcase.htm)**: IfcBuildingElement with the following parameters. Only straight walls are reconstructed in this repo. As such, only coplanar elements can contribute to the parameter estimation. The orthogonal surfaces will be used to define [IfcOpeningElement](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcproductextension/lexical/ifcopeningelement.htm) in T10.
    - IfcLocalPlacement (m): the two control points at both ends of the wall axis. Note that the wall axis is at the center of the wall. 
    - Wall Thickness (m): uniform distance between both wall faces.
    - base constraint (URI): bottom reference level
    - base offset (m): offset from the base constraint level to the bottom of the IfcBuildingElement
    - top constraint (URI): top reference level
    - top offset (m): offset from the base constraint level to the bottom of the IfcBuildingElement
    - HasOpenings (URI): links through [IfcRelVoidsElement](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcproductextension/lexical/ifcrelvoidselement.htm) to [IfcOpeningElement](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcproductextension/lexical/ifcopeningelement.htm) objects that define holes in the wall. 
    - resource (Open3D.TriangleMesh): OrientedBoundingBox of the wall

-  **T7. [IfcColumn](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcsharedbldgelements/lexical/ifccolumn.htm)**:  IfcBuildingElement with the following parameters.
    - IfcCircleProfileDef.Radius (m): radius or w,h
    - IfcLocalPlacement (m): center of column at the base of the column
    - base constraint (URI): bottom reference level
    - base offset (m): offset from the base constraint level to the bottom of the IfcBuildingElement
    - top constraint (URI): top reference level
    - top offset (m): offset from the top constraint level to the top of the IfcBuildingElement
    - resource (Open3D.TriangleMesh): cylinder or orientedBoundingBox

-  **T8. [IfcDoor](https://standards.buildingsmart.org/IFC/DEV/IFC4_2/FINAL/HTML/schema/ifcsharedbldgelements/lexical/ifcdoor.htm)**: Secondary IfcBuildingElement with the following parameters.
    - width (m): w
    - height (m): h
    - IfcLocalPlacement (m): center of the door
    - wall constraint (URI): link to reference wall
    - resource (Open3D.TriangleMesh): OrientedBoundingBox of the door

-  **T9. [IfcSpace](https://standards.buildingsmart.org/IFC/DEV/IFC4_2/FINAL/HTML/schema/ifcproductextension/lexical/ifcspace.htm)**: Non-metric element that is defined based on its bounding elements. 
    - BoundedBy (URI): link to slab and wall elements
    - IfcBuildingStorey (URI): link to reference level
    - resource (Open3D.TriangleMesh): OrientedBoundingBox of the space (slab to slab)

-  **T10. [IfcOpeningElement](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcproductextension/lexical/ifcopeningelement.htm)**: Child element of [IfcWallStandardCase](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcsharedbldgelements/lexical/ifcwallstandardcase.htm) that defines a boolean operation of subtraction between the geometric bodies of the element and the opening. IfcBuildingElement with the following parameters.
    - Geometry : The easiest definition is an orientedBoundingBox orthogonal to the wall's axis. This geometry is defined by its parameters (c_x,c_y,c_z,R_x,R_y,R_z,s_u,s_v,s_w) or it's 8 bounding points.
    - resource (Open3D.TriangleMesh): OrientedBoundingBox of the opening
