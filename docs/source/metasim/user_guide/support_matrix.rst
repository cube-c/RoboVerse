Support Matrix
==============

This page provides a matrix of the support for different simulators in MetaSim.

Simulation Parameters
---------------------

The following table shows the parameters that can be set for the simulation in each simulator. ``✓`` means the parameter is supported. Empty cell means the parameter is not supported. Other values means the default value in the original simulator.

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20 20 20

   * - Parameter
     - IsaacLab
     - IsaacGym
     - MuJoCo
     - Genesis
     - SAPIEN v3
     - PyBullet
   * - ``dt``
     - `1/60 <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html#isaaclab.sim.SimulationCfg.dt>`_
     - `1/60 <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?highlight=substeps#isaacgym.gymapi.SimParams.substeps>`_
     - `1/500 <https://mujoco.readthedocs.io/en/stable/XMLreference.html#option>`_
     - `1/100 <https://genesis-world.readthedocs.io/en/latest/api_reference/scene/simulator.html#genesis.engine.simulator.Simulator.dt>`_
     - 1/100
     - `1/240 <https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit?tab=t.0#heading=h.kyqqrtg5v8nc>`_
   * - ``solver_type``
     - `✓ <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html#isaaclab.sim.PhysxCfg.solver_type>`_
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html#isaacgym.gymapi.PhysXParams.solver_type>`_
     -
     -
     -
     -

Physics Engine Parameters
-------------------------

The following table shows the parameters that can be set for the physics engine in each simulator.

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20 20 20

   * - Parameter
     - IsaacLab
     - IsaacGym
     - MuJoCo
     - Genesis
     - SAPIEN v3
     - PyBullet
   * - ``bounce_threshold_velocity``
     - `✓ <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html#isaaclab.sim.PhysxCfg.bounce_threshold_velocity>`_
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?highlight=bounce_threshold_velocity#isaacgym.gymapi.PhysXParams.bounce_threshold_velocity>`_
     -
     -
     -
     -
   * - ``contact_offset``
     -
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?#isaacgym.gymapi.RigidShapeProperties.contact_offset>`_
     -
     -
     -
     -
   * - ``friction_correlation_distance``
     - `✓ <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html#isaaclab.sim.PhysxCfg.friction_correlation_distance>`_
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?highlight=friction_correlation_distance#isaacgym.gymapi.PhysXParams.friction_correlation_distance>`_
     -
     -
     -
     -
   * - ``friction_offset_threshold``
     - `✓ <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html#isaaclab.sim.PhysxCfg.friction_offset_threshold>`_
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?highlight=friction_correlation_distance#isaacgym.gymapi.PhysXParams.friction_offset_threshold>`_
     -
     -
     -
     -
   * - ``num_position_iterations``
     -
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?#isaacgym.gymapi.PhysXParams.num_position_iterations>`_
     -
     -
     -
     -
   * - ``num_velocity_iterations``
     -
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?#isaacgym.gymapi.PhysXParams.num_position_iterations>`_
     -
     -
     -
     -
   * - ``rest_offset``
     -
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?highlight=rest_offset#isaacgym.gymapi.RigidShapeProperties.rest_offset>`_
     -
     -
     -
     -

Resource Parameters
--------------------

The following table shows the parameters related to resource management in each simulator.

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20 20 20

   * - Parameter
     - IsaacLab
     - IsaacGym
     - MuJoCo
     - Genesis
     - SAPIEN v3
     - PyBullet
   * - ``num_threads``
     -
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html#isaacgym.gymapi.PhysXParams.num_threads>`_
     -
     -
     -
     -

Misc Parameters
---------------

The following table shows the parameters that are not categorized in the above tables in each simulator.

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20 20 20

   * - Parameter
     - IsaacLab
     - IsaacGym
     - MuJoCo
     - Genesis
     - SAPIEN v3
     - PyBullet
   * - ``replace_cylinder_with_capsule``
     -
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html#isaacgym.gymapi.AssetOptions.replace_cylinder_with_capsule>`_
     -
     -
     -
     -
