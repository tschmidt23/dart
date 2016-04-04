dart: Dense Articulated Real-time Tracking
=======

dart is a C++ library for tracking arbitrary articulated models with an RGB-D 
camera. It achieves real-time performance with the aid of a highly parallel CUDA 
implementation and state-of-the-art GPUs.

Required Dependencies
---------------------

**CUDA:** https://developer.nvidia.com/cuda-zone

**Eigen 3:** sudo apt-get install libeigen3-dev

**GNU libmatheval:** sudo apt-get install libmatheval-dev

**libjpeg:** sudo apt-get install libjpeg-dev

**libpng:** sudo apt-get install libpng-dev

**tinyxml:** sudo apt-get install libtinyxml-dev

Optional Dependencies
---------------------

**Pangolin [GUI for the examples]:** https://github.com/stevenlovegrove/Pangolin

**OpenNI [for PrimeSense sensors]:** https://github.com/OpenNI/OpenNI2 or http://structure.io/openni

**DepthSense SDK [for Intel sensor]:** sudo apt-get install depthsensesdk

**Open Asset Import Library:** sudo apt-get install libassimp-dev [for mesh models]

**gtest [for testing]:** sudo apt-get install libgtest-dev; cd /usr/src/gtest; sudo mkdir build; cd build; sudo cmake ..; sudo make; sudo mv libgtest* /usr/lib/;

Installation
------------

cd [dart directory]
mkdir build
cd build
cmake ..
make

Example usage
------------

An example demonstrating the use of DART to track robot hands manipulating objects, including the depth and color video, can be downloaded from [here](http://homes.cs.washington.edu/~tws10/dartExample.tar.gz).

Notes on using the library
------------

- OpenGL context: dart is a very visual library and therefore assumes that it 
will be used in collaboration with a GUI tool. If an instance of dart::Tracker 
is instantiated with no active OpenGL context, it will not work. If you are not 
using the library with a GUI, you will have to create an OpenGL context (e.g. by 
using glutCreateWindow to create a 1x1 window).

- Frames vs. Joints vs. SDFs: These are three separate but related ways in which
parts of a DART model can be referenced. A frame is a frame of reference in the
kinematic chain, a joint is a connection between two frames with a single degree
of freedom, and a signed distance function (SDF) implicitly stores all geometry
attached to a single frame. Every model has at least one frame (the root) but
need not have any joints or signed distance functions. Because loops are not
allowed, a model with N joints will have N+1 frames (and N+6 degrees of 
freedom). The number of SDFs is at most equal to the number of frames, but may
be less if there are frames with no geometry attached to them. Functions in the
Model class and subclasses that require indexing part of the model will indicate
in the parameter name whether the index is by joint, by frame, or by SDF.

Model file format
------------

DART models are stored as XML files which define the kinematic and geometric 
structure, optionally reference other mesh files to further describe the 
geometry. All models open with the "model" tag which has a single attribute, 
"version" describing the version of the DART XML format (current 1), like so:

    <model version ="1">
      [model here]
    </model>

The model can then optionally specify a number of parameters using the "param" 
tag, with attributes "name" (string) and "value" (floating point), like so:

    <param name="armLength" value="1.5"/>

These parameters can then be referenced when defining sizes, positions, or 
orientations, as described below. Parameters are useful if the same value 
appears multiple times in your specification (as is often the case) or if you 
would like to set the parameter values programatically.

After defining parameters, the model may contain a number of hierachically 
nested "frame" and "geom" tags, which specify new rigid body frames of reference 
or geometric objects, respectively. The "frame" tag requires four attributes,
"jointName" (string), "jointType" (currently accepts either "rotational" or
"prismatic", "jointMin" (floating point), and "jointMax" (floating point), the
last two of which define the joint limits. Additionally, the frame tag requires
three nested tags, "position", "orientation", and "axis", each of which require
three floating point attributes, "x", "y", and "z". An example might look like
this:

	<frame jointName="leftElbow" jointType="rotational" jointMin="0" jointMax="3.1416">
	    <position x="0" y="0" z="1.5" />
	    <orientation x="0" y="0" z="1.5708" />
	    <axis x="1" y="0" z="0"/>
	    [frame children here]
	</frame>

This snippet defines a new frame of reference relative to its parent (the XML
node directly above it in the hierarchy, or the root if the parent is the 
"model" tag). The transform from this frame of reference to the world is given
by:

T_w,f = T_w,p*Trans*R_z*R_y*R_x*R_axis(theta)

where T_w,p gives the transform from the parent to the world, Trans is a
translation-only transform given by the "position" tag, R_z, R_y, and R_x are
rotations about the z, y, and x axes (i.e. Euler angles) given by the
corresponding entries in the "orientation" tag, and R_axis is a rotation by
theta around the axis defined by the "axis" tag, with theta being given by the
articulated pose of the model.

Finally, geometry can be rigidly attached to any frame of reference in the model
by nesting a "geom" tag within a "frame" tag (or within the "model" tag for root
geometry). The geometry tag requires 13 attributes: "type" (currently accepts
"sphere","cylinder","cube", or "mesh"), "sx", "sy", and "sz", which define the
scaling of the geometry, "tx", "ty", and "tz", which define the translation of
the geometry root relative to the rigid body frame of reference, "rx", "ry" and
"rz", which define the orientation relative to the rigid body frame of reference
(also in Euler angles, as with the "frame" tag), and "red", "green" and "blue",
which define the geometry color, which is not used for tracking but will affect
how the model is rendered for debugging purposes. If "type" is set to "mesh",
there is one final attribute, "meshFile", which gives the location of the mesh
file, relative to the location of the XML file.

