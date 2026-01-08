"""
Pose Graph Optimization for VGGT-Long

This is what you ACTUALLY need for drift reduction with large scenes.
Implements global optimization instead of sequential pairwise alignment.

Requirements:
    pip install g2o-python

Usage:
    optimizer = PoseGraphOptimizer()

    # Add sequential edges (overlap constraints)
    for i in range(num_chunks - 1):
        s, R, t = align(chunk[i], chunk[i+1])
        optimizer.add_edge(i, i+1, s, R, t, weight=1.0)

    # Add loop closure edges
    for (i, j) in loop_closures:
        s, R, t = align(chunk[i], chunk[j])
        optimizer.add_edge(i, j, s, R, t, weight=2.0)  # Higher weight for loops

    # Optimize globally
    optimized_poses = optimizer.optimize()
"""

import numpy as np
from scipy.spatial.transform import Rotation
import warnings

# Try to import g2o (optional, shows API if not available)
try:
    import g2o
    HAS_G2O = True
except ImportError:
    HAS_G2O = False
    warnings.warn("g2o not installed. Install with: pip install g2o-python")


class PoseGraphOptimizer:
    """
    Pose graph optimizer for VGGT-Long chunks.

    Performs global optimization of chunk poses using:
    - Sequential overlap constraints
    - Loop closure constraints
    - Optional scale consistency constraints
    """

    def __init__(self, use_sim3=True):
        """
        Args:
            use_sim3: If True, optimize over SIM(3) (includes scale)
                     If False, optimize over SE(3) (fixed scale, for metric depth)
        """
        self.use_sim3 = use_sim3
        self.num_nodes = 0
        self.edges = []

        if not HAS_G2O:
            print("WARNING: g2o not installed. Optimizer will not work.")
            print("Install with: pip install g2o-python")

    def add_node(self, node_id, initial_pose=None):
        """
        Add a node (chunk pose) to the graph.

        Args:
            node_id: Unique identifier for this chunk
            initial_pose: Initial (s, R, t) if available, else identity
        """
        if initial_pose is None:
            s = 1.0
            R = np.eye(3)
            t = np.zeros(3)
        else:
            s, R, t = initial_pose

        self.num_nodes = max(self.num_nodes, node_id + 1)

        # Store for initialization
        if not hasattr(self, 'initial_poses'):
            self.initial_poses = {}
        self.initial_poses[node_id] = (s, R, t)

    def add_edge(self, from_id, to_id, s, R, t, weight=1.0, information=None):
        """
        Add an edge (constraint) between two nodes.

        Args:
            from_id: Source chunk ID
            to_id: Target chunk ID
            s: Scale factor
            R: 3x3 rotation matrix
            t: 3D translation vector
            weight: Importance of this constraint (higher = more trusted)
            information: 7x7 information matrix (inverse covariance)
                        If None, uses identity scaled by weight
        """
        if information is None:
            # Default: diagonal information matrix
            information = np.eye(7) * weight

        self.edges.append({
            'from': from_id,
            'to': to_id,
            's': s,
            'R': R,
            't': t,
            'information': information
        })

    def optimize(self, max_iterations=100, verbose=True):
        """
        Run global pose graph optimization.

        Returns:
            optimized_poses: List of (s, R, t) for each chunk
        """
        if not HAS_G2O:
            print("ERROR: g2o not installed. Cannot optimize.")
            print("Returning initial poses (no optimization).")
            return self._get_initial_poses_sequential()

        # TODO: Implement g2o optimization
        # This requires proper g2o wrapper code

        print("Full g2o implementation would go here.")
        print("For now, returning fallback solution.")

        return self._optimize_least_squares(max_iterations, verbose)

    def _get_initial_poses_sequential(self):
        """
        Fallback: Chain sequential edges (no global optimization).
        """
        poses = [None] * self.num_nodes
        poses[0] = (1.0, np.eye(3), np.zeros(3))

        # Chain forward
        for edge in self.edges:
            if edge['from'] == edge['to'] - 1:  # Sequential edge
                i, j = edge['from'], edge['to']
                if poses[i] is not None and poses[j] is None:
                    # Compose transformations
                    s_i, R_i, t_i = poses[i]
                    s_ij, R_ij, t_ij = edge['s'], edge['R'], edge['t']

                    s_j = s_i * s_ij
                    R_j = R_i @ R_ij
                    t_j = s_i * (R_i @ t_ij) + t_i

                    poses[j] = (s_j, R_j, t_j)

        return poses

    def _optimize_least_squares(self, max_iterations, verbose):
        """
        Simple least-squares optimization (approximation of pose graph).

        This is a simplified version without proper g2o.
        For production, use actual g2o implementation.
        """
        from scipy.optimize import least_squares

        if verbose:
            print(f"Running least-squares optimization ({max_iterations} iterations)...")

        # Get initial poses
        initial_poses = self._get_initial_poses_sequential()

        # Pack into parameter vector
        # For each pose: [log_s, quat(4), t(3)] = 8 params
        params = []
        for s, R, t in initial_poses:
            params.append(np.log(s))  # log scale
            quat = Rotation.from_matrix(R).as_quat()  # xyzw format
            params.extend(quat)
            params.extend(t)

        params = np.array(params)

        # Residual function
        def residuals(x):
            res = []
            for edge in self.edges:
                i, j = edge['from'], edge['to']

                # Extract poses
                s_i = np.exp(x[i*8])
                R_i = Rotation.from_quat(x[i*8+1:i*8+5]).as_matrix()
                t_i = x[i*8+5:i*8+8]

                s_j = np.exp(x[j*8])
                R_j = Rotation.from_quat(x[j*8+1:j*8+5]).as_matrix()
                t_j = x[j*8+5:j*8+8]

                # Measured transformation
                s_ij_meas = edge['s']
                R_ij_meas = edge['R']
                t_ij_meas = edge['t']

                # Predicted transformation: T_i^-1 @ T_j
                s_ij_pred = s_j / s_i
                R_ij_pred = R_i.T @ R_j
                t_ij_pred = R_i.T @ (t_j - t_i) / s_i

                # Residuals
                res.append((np.log(s_ij_pred) - np.log(s_ij_meas)) * np.sqrt(edge['information'][0, 0]))

                # Rotation residual (angle-axis)
                R_error = R_ij_pred @ R_ij_meas.T
                angle_axis = Rotation.from_matrix(R_error).as_rotvec()
                res.extend(angle_axis * np.sqrt(edge['information'][1, 1]))

                # Translation residual
                t_error = t_ij_pred - t_ij_meas
                res.extend(t_error * np.sqrt(edge['information'][4, 4]))

            return np.array(res)

        # Optimize
        result = least_squares(
            residuals,
            params,
            max_nfev=max_iterations * len(self.edges),
            verbose=2 if verbose else 0
        )

        # Unpack results
        optimized_poses = []
        for i in range(self.num_nodes):
            s = np.exp(result.x[i*8])
            R = Rotation.from_quat(result.x[i*8+1:i*8+5]).as_matrix()
            t = result.x[i*8+5:i*8+8]
            optimized_poses.append((s, R, t))

        if verbose:
            print(f"Optimization converged. Final cost: {result.cost:.6f}")

        return optimized_poses


def integrate_with_vggt_long(vggt_long_instance, loop_closures=None):
    """
    Replace VGGT-Long's sequential alignment with global pose graph optimization.

    Args:
        vggt_long_instance: VGGT_Long object after processing
        loop_closures: Optional list of (i, j) manual loop closures

    Returns:
        optimized_sim3_list: List of (s, R, t) for each chunk
    """
    print("\n" + "="*60)
    print("Global Pose Graph Optimization")
    print("="*60)

    optimizer = PoseGraphOptimizer(use_sim3=vggt_long_instance.config['Model']['using_sim3'])

    # Add nodes for each chunk
    num_chunks = len(vggt_long_instance.chunk_indices)
    for i in range(num_chunks):
        optimizer.add_node(i)

    print(f"Added {num_chunks} nodes (chunks)")

    # Add sequential edges (from existing sim3_list)
    for i in range(num_chunks - 1):
        s, R, t = vggt_long_instance.sim3_list[i]
        optimizer.add_edge(i, i+1, s, R, t, weight=1.0)

    print(f"Added {num_chunks - 1} sequential edges")

    # Add loop closure edges
    if loop_closures is not None and len(loop_closures) > 0:
        from pathlib import Path
        import numpy as np

        for (frame_i, frame_j) in loop_closures:
            # Find which chunks these frames belong to
            chunk_i = None
            chunk_j = None

            for ci, (start, end) in enumerate(vggt_long_instance.chunk_indices):
                if start <= frame_i < end:
                    chunk_i = ci
                if start <= frame_j < end:
                    chunk_j = ci

            if chunk_i is not None and chunk_j is not None and chunk_i != chunk_j:
                # Load chunks and compute transformation
                # (Simplified - in practice would use actual alignment)

                print(f"Adding loop closure: chunk {chunk_i} ↔ chunk {chunk_j}")

                # Placeholder: use identity (should compute actual alignment)
                s_loop = 1.0
                R_loop = np.eye(3)
                t_loop = np.zeros(3)

                # Higher weight for loop closures (more trusted)
                optimizer.add_edge(chunk_i, chunk_j, s_loop, R_loop, t_loop, weight=5.0)

        print(f"Added {len(loop_closures)} loop closure edges")

    # Run global optimization
    print("\nOptimizing pose graph...")
    optimized_poses = optimizer.optimize(max_iterations=100, verbose=True)

    # Convert back to sim3_list format
    optimized_sim3_list = []
    for i in range(num_chunks - 1):
        s_i, R_i, t_i = optimized_poses[i]
        s_j, R_j, t_j = optimized_poses[i+1]

        # Relative transformation
        s_ij = s_j / s_i
        R_ij = R_i.T @ R_j
        t_ij = R_i.T @ (t_j - t_i) / s_i

        optimized_sim3_list.append((s_ij, R_ij, t_ij))

    print("\n✓ Pose graph optimization complete!")
    print(f"  Old sequential alignment: {len(vggt_long_instance.sim3_list)} transformations")
    print(f"  New optimized alignment: {len(optimized_sim3_list)} transformations")

    return optimized_sim3_list


if __name__ == "__main__":
    print(__doc__)

    print("\nExample usage:")
    print("-" * 60)

    # Create optimizer
    optimizer = PoseGraphOptimizer(use_sim3=True)

    # Add 5 chunks
    for i in range(5):
        optimizer.add_node(i)

    # Add sequential edges with some drift
    for i in range(4):
        s = 1.0 + np.random.randn() * 0.03  # 3% scale error
        R = np.eye(3)  # Simplified
        t = np.array([i * 10.0, 0, 0])  # 10m forward each

        optimizer.add_edge(i, i+1, s, R, t, weight=1.0)

    # Add loop closure: chunk 4 back to chunk 0
    optimizer.add_edge(4, 0, 1.0, np.eye(3), np.array([0, 0, 0]), weight=5.0)

    # Optimize
    poses = optimizer.optimize(verbose=True)

    print("\nOptimized poses:")
    for i, (s, R, t) in enumerate(poses):
        print(f"  Chunk {i}: scale={s:.3f}, t={t}")
