# Scan-to-BIM-CVPR-2024
<div style="text-align: justify;">
This is the KUL and FBK repo for the [4th International Scan-to-BIM competition](https://cv4aec.github.io/) to reconstruct wall, column and door elements from point cloud data in public buildings. We split the process into three steps, a <strong>Preprocessing</strong> step (task 0), a <strong>Detection</strong> step (tasks 1-4) and a <strong>Reconstruction</strong> (tasks 5-9) step. In total, 10 tasks are defined. Each step is explained below.
</div>

![Preprocessing result showing submeshes for each wall, column, and door element.](docs/assets/t6_walls2.PNG)

<p align="center"><i>Figure 1: Preprocessing result showing submeshes for each wall, column, and door element.</i></p>

## Preprocessing
<div style="text-align: justify;">
The preprocessing includes subsampling the point cloud to 0.01m and parsing the training data jsons to triangle mesh objects. The result is an .obj with submeshes for each wall, column and door element (see Figure 1). Additionally, we segment the point clouds according to these elements to form the ground truth for the instance segmentation training (Figure 2). We also generate RDF graphs with metric metadata of each point cluster and link them to the BIM Objects so they can be tracked throughout the reconstruction process. Take a look at our [GEOMAPI](https://ku-leuven-geomatics.github.io/geomapi/) toolbox to read more about how this is done.
</div>

<div style="display: flex; justify-content: space-around;">
    <div style="flex: 1; text-align: center;">
        <img src="docs/assets/35_Lab_02_F1_small1_t6_2.png" alt="35 Lab 02 F1 Small1 T6 2" style="width: 100%; height: 100%; object-fit: cover;">
        <p>Figure 2: Segmented Point Cloud</p>
    </div>
    <div style="flex: 1; text-align: center;">
        <img src="docs/assets/rdf1.png" alt="RDF1" style="width: 100%; height: 100%; object-fit: cover;">
        <p>Figure 3: RDF graph with Metadata properties</p>
    </div>
</div>



## Detection
In the first step, we compute the instance segmentation of the primary (walls, ceilings, floors, columns) and secondary structure classes ( doors, windows). Two scalar field are assigned to the unstructured point clouds. First, a **class label** is computed for every point of the in total 6 classes (0.Floors, 1.Ceilings, 2. Walls, 3.Columns, 4.Doors, 255.Unassigned). Second, an **object label** is assigned to every point and a json is computed with the 3D information of the detected objects. 
![Alt text](/docs/assets/detection.PNG "detection")

- **[T1. Semantic Segmentation](./scripts/t1_semantic_segmentation.ipynb)**: [PTV3+PPT](https://github.com/Pointcept/PointTransformerV3) is an excellent baseline model for unstructured points clouds such as walls and column. However, it doesn't do instance segmentation and thus a clustering must be implemented to achieve an instance segmentation. 
    

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

In the second step, we compute the parametric information and geometries of the BIM elements. Per convention, BIM models are hierarchically reconstructed starting from the [IfcBuildingStory](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcproductextension/lexical/ifcbuildingstorey.htm) elements, followed by the [IfcWallStandardCase](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcsharedbldgelements/lexical/ifcwallstandardcase.htm) and [IfcColumn](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcsharedbldgelements/lexical/ifccolumn.htm) elements. Once the primary building elements are established, the secondary building elements ([IfcDoor](https://standards.buildingsmart.org/IFC/DEV/IFC4_2/FINAL/HTML/schema/ifcsharedbldgelements/lexical/ifcdoor.htm)), non-metric elements ([IfcSpace](https://standards.buildingsmart.org/IFC/DEV/IFC4_2/FINAL/HTML/schema/ifcproductextension/lexical/ifcspace.htm)) and wall detailing ([IfcOpeningElement](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcproductextension/lexical/ifcopeningelement.htm)). To this end, a scene Graph is constructed that links together the different elements. However, as the competition requires very specific geometries, we will also generate the necassary geometry for the competition aswell. 


-  **T5. [IfcBuildingStory](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcproductextension/lexical/ifcbuildingstorey.htm)**: For the reference levels, we consider the architectural levels (Finish Floor Level or FFL) since we are modeling the visible construction elements in the architectural domain.
    - IfcLocalPlacement (m): center point of the IfcBuildingElement
    - FootPrint (m): 2D lineset or parametric 2D orientedBoundingBox (c_x,c_y,c_z,R_z,s_u,s_v)
    - Elevation (m): c_z
    - Resource (Open3D.TriangleMesh): plane of the storey

-  **T6. [IfcWallStandardCase](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcsharedbldgelements/lexical/ifcwallstandardcase.htm)**: IfcBuildingElement with the following parameters. Only straight walls are reconstructed in this repo (because they are the only type of wall in the challenge). As such, only coplanar elements can contribute to the parameter estimation. The orthogonal surfaces will be used to define [IfcOpeningElement](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcproductextension/lexical/ifcopeningelement.htm) elements in T10. Note that single-faced walls will be gived a default thickness of 0.1m, according to standard indoor wall. Additionally, the wall thickness will not be clustered (which is typical in a scan-to-bim project), to achieve the highest possible accuracy. 
    - IfcLocalPlacement (p_1,p_2): the two control points at both ends of the wall axis. Note that the wall axis is at the center of the wall. 
    - Wall Thickness (m): uniform distance between both wall faces.
    - base constraint (URI): bottom reference level
    - base offset (m): offset from the base constraint level to the bottom of the IfcBuildingElement
    - top constraint (URI): top reference level
    - top offset (m): offset from the base constraint level to the bottom of the IfcBuildingElement
    - HasOpenings (URI): links through [IfcRelVoidsElement](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcproductextension/lexical/ifcrelvoidselement.htm) to [IfcOpeningElement](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcproductextension/lexical/ifcopeningelement.htm) objects that define holes in the wall. 
    - Resource (Open3D.TriangleMesh): OrientedBoundingBox of the wall

-  **T7. [IfcColumn](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcsharedbldgelements/lexical/ifccolumn.htm)**:  IfcBuildingElement with the following parameters.
    - IfcCircleProfileDef.Radius (m): radius or w,h
    - IfcLocalPlacement (c): center of column at the base of the column
    - base constraint (URI): bottom reference level
    - base offset (m): offset from the base constraint level to the bottom of the IfcBuildingElement
    - top constraint (URI): top reference level
    - top offset (m): offset from the top constraint level to the top of the IfcBuildingElement
    - Resource (Open3D.TriangleMesh): cylinder or orientedBoundingBox

-  **T8. [IfcDoor](https://standards.buildingsmart.org/IFC/DEV/IFC4_2/FINAL/HTML/schema/ifcsharedbldgelements/lexical/ifcdoor.htm)**: Secondary IfcBuildingElement with the following parameters.
    - width (m): w
    - height (m): h
    - IfcLocalPlacement (m): center of the door
    - wall constraint (URI): link to reference wall
    - resource (Open3D.TriangleMesh): OrientedBoundingBox of the door

-  **T9. [IfcSpace](https://standards.buildingsmart.org/IFC/DEV/IFC4_2/FINAL/HTML/schema/ifcproductextension/lexical/ifcspace.htm)**: Non-metric element that is defined based on its bounding elements. 
    - BoundedBy (URI): link to slab and wall elements
    - IfcBuildingStorey (URI): link to reference level
    - Resource (Open3D.TriangleMesh): OrientedBoundingBox of the space (slab to slab)

-  **T10. [IfcOpeningElement](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcproductextension/lexical/ifcopeningelement.htm)**: These are child elements of [IfcWallStandardCase](https://standards.buildingsmart.org/IFC/RELEASE/IFC2x3/TC1/HTML/ifcsharedbldgelements/lexical/ifcwallstandardcase.htm) to increase the detailing of the initial wall geometry. They define boolean subtraction operations between both geometric bodies of the element and the opening. It has the following parameters.
    - Geometry (c_x,c_y,c_z,R_x,R_y,R_z,s_u,s_v,s_w) : The easiest definition is an orientedBoundingBox orthogonal to the wall's axis. This geometry is defined by its parameters (c_x,c_y,c_z,R_x,R_y,R_z,s_u,s_v,s_w) or it's 8 bounding points.
    - Resource (Open3D.OrientedBoundingBox): OrientedBoundingBox of the opening
    - HasOpenings (Inverse IfcRelVoidsElement URI)
