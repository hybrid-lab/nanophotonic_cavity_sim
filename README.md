# nanophotonic_cavity

This repository has two main simulation pipelines.

The first pipeline is for band structure simulations using MPB, implemented in `bandstructure_class.py`. The `bandstructure_tutorial` explains how to use this class.

The second pipeline is for cavity simulations using Tidy3D. The `cavity_tutorial` explains how to use the different classes to simulate various cavity properties in Tidy3D.

The cavity simulation workflow is organized around a `cavity` class, which generates the layouts and assembles the different parts of the cavity by calling the corresponding component classes, such as `taper`, `mirror`, and `defect`. It also includes a function to generate GDS files, handled through the layout repository of MJ's group.

The `hole.py` file defines the hole geometry used in the unit cells. To add a new geometry, it is sufficient to modify the `hole_polygon_2d` function.

Once a `cavity` class has been instantiated for a specific design, a `cavity_simulation` class can be created. This class handles all aspects of the Tidy3D simulations.

Finally, `general_GDS_patterns.py` is a utility file used to generate various GDS patterns, such as alignment markers and other simple geometries.
