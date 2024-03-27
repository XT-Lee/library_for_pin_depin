import points_analysis_2D as pa
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import distance
RT = math.sqrt(3)
# this file is archimedean_module


class archimedean_tilings:
    def __init__(self):
        R"""
        example_code:
            import workflow_analysis as wa
            import matplotlib.pyplot as plt
            at = wa.archimedean_tilings()
            at.generate_type10_part()
            points = at.generate_lattices(6)
            dula = at.get_dual_lattice(points)
            fig,ax = plt.subplots()
            ax.scatter(points[:,0],points[:,1],color='k')
            ax.scatter(dula[:,0],dula[:,1],facecolors='none',edgecolors='k')#,marker = 'x'
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
        """
        pass

    def generate_type11(self, a=1):
        R"""
        Introduction:
            (3^3,4^2) squares and triangles.
            a: the edge length of a single tile.
            n: n*n lattices to generate. 
        example:
            import workflow_analysis as wa
            import matplotlib.pyplot as plt
            at = wa.archimedean_tilings()
            at.generate_type11(2)
            points = at.generate_lattices([12,3])
            fig,ax = plt.subplots()
            ax.scatter(points[:,0],points[:,1])
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
        source:
            def generate_honeycomb(self,a,n):
            uc = hoomd.lattice.unitcell(N=4,
                                    a1=[3*a, 0, 0],
                                    a2=[0, math.sqrt(3)*a, 0],
                                    a3=[0, 0, 1],
                                    dimensions=2,
                                    position=[[0,0,0], [1/2*a, math.sqrt(3)/2*a, 0],[3/2*a,math.sqrt(3)/2*a,0],[2*a,0,0]],
                                    type_name=['A', 'A','A','A'],
                                    diameter=[1,1,1,1]);
            self.system = hoomd.init.create_lattice(unitcell=uc, n=n);
        """
        rt = RT
        # N=4
        self.a1 = np.array([1, 0, 0])*a
        self.a2 = np.array([0, 2+rt, 0])*a
        self.a3 = np.array([0, 0, 0])
        # dimensions=2
        self.position = np.array(
            [[0, 2 + 0.5 * rt, 0],
             [0, 1 + 0.5 * rt, 0],
             [0.5, 1, 0],
             [0.5, 0, 0]]) * a  # [0.5,2+rt,0],

    def generate_type11_part(self, a=1):
        R"""
        Introduction:
            (3^3,4^2) squares and triangles.
            a: the edge length of a single tile.
            n: n*n lattices to generate. 
        example:
            import workflow_analysis as wa
            import matplotlib.pyplot as plt
            at = wa.archimedean_tilings()
            at.generate_type11_part(2)
            points = at.generate_lattices([12,3])
            fig,ax = plt.subplots()
            ax.scatter(points[:,0],points[:,1])
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
        source:
            def generate_honeycomb(self,a,n):
            uc = hoomd.lattice.unitcell(N=4,
                                    a1=[3*a, 0, 0],
                                    a2=[0, math.sqrt(3)*a, 0],
                                    a3=[0, 0, 1],
                                    dimensions=2,
                                    position=[[0,0,0], [1/2*a, math.sqrt(3)/2*a, 0],[3/2*a,math.sqrt(3)/2*a,0],[2*a,0,0]],
                                    type_name=['A', 'A','A','A'],
                                    diameter=[1,1,1,1]);
            self.system = hoomd.init.create_lattice(unitcell=uc, n=n);
        """
        rt = RT
        # N=2
        self.a1 = np.array([1, 0, 0])*a
        self.a2 = np.array([0, 2+rt, 0])*a
        self.a3 = np.array([0, 0, 0])
        # dimensions=2
        self.position = np.array([[0, 2+0.5*rt, 0], [0.5, 1, 0]])*a  # [0.5,2+rt,0],

    def generate_type10(self, a=1):
        R"""
        Introduction:
            (3^2,4,3,4) squares and triangles.
            a: the edge length of a single tile.
            n: n*n lattices to generate. 
        example:
            import workflow_analysis as wa
            import matplotlib.pyplot as plt
            at = wa.archimedean_tilings()
            at.generate_type11(2)
            points = at.generate_lattices([12,3])
            fig,ax = plt.subplots()
            ax.scatter(points[:,0],points[:,1])
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
        source:
            def generate_honeycomb(self,a,n):
            uc = hoomd.lattice.unitcell(N=4,
                                    a1=[3*a, 0, 0],
                                    a2=[0, math.sqrt(3)*a, 0],
                                    a3=[0, 0, 1],
                                    dimensions=2,
                                    position=[[0,0,0], [1/2*a, math.sqrt(3)/2*a, 0],[3/2*a,math.sqrt(3)/2*a,0],[2*a,0,0]],
                                    type_name=['A', 'A','A','A'],
                                    diameter=[1,1,1,1]);
            self.system = hoomd.init.create_lattice(unitcell=uc, n=n);
        """
        rt = RT
        # N=8
        self.a1 = np.array([1+rt, 0, 0])*a
        self.a2 = np.array([0, 1+rt, 0])*a
        self.a3 = np.array([0, 0, 0])
        # dimensions=2
        self.position = np.array([[0.5+0.5*rt, 0.5+0.5*rt, 0], [0.5+0.5*rt, 1.5+0.5*rt, 0],
                                  [0.5, 1+0.5*rt, 0], [0.5+rt, 1+0.5*rt, 0],
                                  [0.5*rt, 0.5, 0], [1+0.5*rt, 0.5, 0],
                                  [0, 0, 0], [0, 1, 0]])*a

    def generate_type10_part(self, a=1):
        R"""
        Introduction:
            (3^2,4,3,4) squares and triangles.
            a: the edge length of a single tile.
            n: n*n lattices to generate. 
        example:
            import workflow_analysis as wa
            import matplotlib.pyplot as plt
            at = wa.archimedean_tilings()
            at.generate_type11(2)
            points = at.generate_lattices([12,3])
            fig,ax = plt.subplots()
            ax.scatter(points[:,0],points[:,1])
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
        source:
            def generate_honeycomb(self,a,n):
            uc = hoomd.lattice.unitcell(N=4,
                                    a1=[3*a, 0, 0],
                                    a2=[0, math.sqrt(3)*a, 0],
                                    a3=[0, 0, 1],
                                    dimensions=2,
                                    position=[[0,0,0], [1/2*a, math.sqrt(3)/2*a, 0],[3/2*a,math.sqrt(3)/2*a,0],[2*a,0,0]],
                                    type_name=['A', 'A','A','A'],
                                    diameter=[1,1,1,1]);
            self.system = hoomd.init.create_lattice(unitcell=uc, n=n);
        """
        rt = RT
        # N=4
        self.a1 = np.array([1+rt, 0, 0])*a
        self.a2 = np.array([0, 1+rt, 0])*a
        self.a3 = np.array([0, 0, 0])
        # dimensions=2
        self.position = np.array([[0.5+0.5*rt, 0.5+0.5*rt, 0], [0.5+0.5*rt, 1.5+0.5*rt, 0],
                                  [0, 0, 0], [0, 1, 0]])*a

    def generate_type9(self, a=1):
        R"""
        Introduction:
            (3^4,6) triangles and hexagons.
            a: the edge length of a single tile.
            n: n*n lattices to generate. 
        example:
            import workflow_analysis as wa
            import matplotlib.pyplot as plt
            at = wa.archimedean_tilings()
            at.generate_type11(2)
            points = at.generate_lattices([12,3])
            fig,ax = plt.subplots()
            ax.scatter(points[:,0],points[:,1])
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
        source:
            def generate_honeycomb(self,a,n):
            uc = hoomd.lattice.unitcell(N=4,
                                    a1=[3*a, 0, 0],
                                    a2=[0, math.sqrt(3)*a, 0],
                                    a3=[0, 0, 1],
                                    dimensions=2,
                                    position=[[0,0,0], [1/2*a, math.sqrt(3)/2*a, 0],[3/2*a,math.sqrt(3)/2*a,0],[2*a,0,0]],
                                    type_name=['A', 'A','A','A'],
                                    diameter=[1,1,1,1]);
            self.system = hoomd.init.create_lattice(unitcell=uc, n=n);
        """
        rt = RT
        # N=6
        self.a1 = np.array([2.5, -0.5*rt, 0])*a
        self.a2 = np.array([-0.5, 1.5*rt, 0])*a  # np.array([2,rt,0])*a
        self.a3 = np.array([0, 0, 0])
        # dimensions=2
        """self.position = np.array([[3.5,0.5*rt,0],
                                  [3,0,0],[2.5,0.5*rt,0],
                                  [0,0,0],[1,0,0],[2,0,0]])*a
        """
        """
                        np.array([[1,rt,0],
                                  [0.5,0.5*rt,0],[2.5,0.5*rt,0],
                                  [0,0,0],[1,0,0],[2,0,0]])*a
        """
        self.position = np.array([[1, rt, 0],
                                  [0, rt, 0], [0.5, 0.5*rt, 0],
                                  [0, 0, 0], [1, 0, 0], [2, 0, 0]])*a

    def generate_type9_part(self, a=1):
        R"""
        Introduction:
            (3^4,6) triangles and hexagons, remove one point per lattice.
            a: the edge length of a single tile.
            n: n*n lattices to generate. 
        example:

        source:
            def generate_honeycomb(self,a,n):
            uc = hoomd.lattice.unitcell(N=4,
                                    a1=[3*a, 0, 0],
                                    a2=[0, math.sqrt(3)*a, 0],
                                    a3=[0, 0, 1],
                                    dimensions=2,
                                    position=[[0,0,0], [1/2*a, math.sqrt(3)/2*a, 0],[3/2*a,math.sqrt(3)/2*a,0],[2*a,0,0]],
                                    type_name=['A', 'A','A','A'],
                                    diameter=[1,1,1,1]);
            self.system = hoomd.init.create_lattice(unitcell=uc, n=n);
        """
        rt = RT
        # N=5
        self.a1 = np.array([2.5, -0.5*rt, 0])*a  # -1.635
        self.a2 = np.array([-0.5, 1.5*rt, 0])*a  # np.array([2,rt,0])*a
        self.a3 = np.array([0, 0, 0])
        # dimensions=2
        """self.position = np.array([[3.5,0.5*rt,0],
                                  [3,0,0],[2.5,0.5*rt,0],
                                  [0,0,0],[1,0,0],[2,0,0]])*a
        """
        """
                        np.array([[1,rt,0],
                                  [0.5,0.5*rt,0],[2.5,0.5*rt,0],
                                  [0,0,0],[1,0,0],[2,0,0]])*a
        """
        self.position = np.array([[1, rt, 0],
                                  [0, rt, 0],
                                  [0, 0, 0], [1, 0, 0], [2, 0, 0]])*a

    def generate_type9_superlattice(self, a=1):
        R"""
        Introduction:
            (3,4,6,4) triangles, squares and hexagons.
            a: the edge length of a single tile.
            n: n*n lattices to generate. 
        example:
            import workflow_analysis as wa
            import matplotlib.pyplot as plt
            at = wa.archimedean_tilings()
            at.generate_type11(2)
            points = at.generate_lattices([12,3])
            fig,ax = plt.subplots()
            ax.scatter(points[:,0],points[:,1])
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
        source:
            def generate_honeycomb(self,a,n):
            uc = hoomd.lattice.unitcell(N=4,
                                    a1=[3*a, 0, 0],
                                    a2=[0, math.sqrt(3)*a, 0],
                                    a3=[0, 0, 1],
                                    dimensions=2,
                                    position=[[0,0,0], [1/2*a, math.sqrt(3)/2*a, 0],[3/2*a,math.sqrt(3)/2*a,0],[2*a,0,0]],
                                    type_name=['A', 'A','A','A'],
                                    diameter=[1,1,1,1]);
            self.system = hoomd.init.create_lattice(unitcell=uc, n=n);
        """
        rt = RT
        # N=20
        self.a1 = np.array([5, -rt, 0])*a  # 1+rt
        self.a2 = np.array([-1, 3*rt, 0])*a
        self.a3 = np.array([0, 0, 0])
        # dimensions=2
        """
        self.position = np.array([[1, rt, 0],
                                  [0, rt, 0], [0.5, 0.5*rt, 0],
                                  [0, 0, 0], [1, 0, 0], [2, 0, 0]])*a
        
        pos_rt = self.position+self.a1*0.5
        pos_lf = self.position+self.a2*0.5
        pos_top_rt = self.position+self.a1*0.5+self.a2*0.5
        self.position = np.concatenate((self.position, pos_rt, pos_lf, pos_top_rt), axis=0)
        """
        self.position = np.array([[1, rt, 0],
                                  [0, rt, 0], [0.5, 0.5*rt, 0],
                                  [0, 0, 0], [1, 0, 0], [2, 0, 0],
                                  [3.5, 0.5*rt, 0],
                                  [2.5, 0.5*rt, 0], [3, 0, 0],
                                  [2.5, -0.5*rt, 0], [3.5, -0.5*rt, 0], [4.5, -0.5*rt, 0],
                                  [0.5, 2.5*rt, 0],
                                  [-0.5, 2.5*rt, 0], [0, 2*rt, 0],
                                  [-0.5, 1.5*rt, 0], [0.5, 1.5*rt, 0], [1.5, 1.5*rt, 0],
                                  [3, 2*rt, 0],
                                  [2, 2*rt, 0], [2.5, 1.5*rt, 0],
                                  [2, rt, 0], [3, rt, 0], [4, rt, 0]])*a

    def generate_type9_part_superlattice(self, a=1):
        R"""
        Introduction:
            (3,4,6,4) triangles, squares and hexagons.
            a: the edge length of a single tile.
            n: n*n lattices to generate. 
        example:
            import workflow_analysis as wa
            import matplotlib.pyplot as plt
            at = wa.archimedean_tilings()
            at.generate_type11(2)
            points = at.generate_lattices([12,3])
            fig,ax = plt.subplots()
            ax.scatter(points[:,0],points[:,1])
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
        source:
            def generate_honeycomb(self,a,n):
            uc = hoomd.lattice.unitcell(N=4,
                                    a1=[3*a, 0, 0],
                                    a2=[0, math.sqrt(3)*a, 0],
                                    a3=[0, 0, 1],
                                    dimensions=2,
                                    position=[[0,0,0], [1/2*a, math.sqrt(3)/2*a, 0],[3/2*a,math.sqrt(3)/2*a,0],[2*a,0,0]],
                                    type_name=['A', 'A','A','A'],
                                    diameter=[1,1,1,1]);
            self.system = hoomd.init.create_lattice(unitcell=uc, n=n);
        """
        rt = RT
        # N=12
        self.a1 = np.array([5, -rt, 0])*a  # 1+rt
        self.a2 = np.array([-1, 3*rt, 0])*a
        self.a3 = np.array([0, 0, 0])
        # dimensions=2
        """
        self.position = np.array([[1, rt, 0],
                                  [0, rt, 0], [0.5, 0.5*rt, 0],
                                  [0, 0, 0], [1, 0, 0], [2, 0, 0],
                                  [3.5, 0.5*rt, 0],
                                  [2.5, 0.5*rt, 0], [3, 0, 0],
                                  [2.5, -0.5*rt, 0], [3.5, -0.5*rt, 0], [4.5, -0.5*rt, 0],
                                  [0.5, 2.5*rt, 0],
                                  [-0.5, 2.5*rt, 0], [0, 2*rt, 0],
                                  [-0.5, 1.5*rt, 0], [0.5, 1.5*rt, 0], [1.5, 1.5*rt, 0],
                                  [3, 2*rt, 0],
                                  [2, 2*rt, 0], [2.5, 1.5*rt, 0],
                                  [2, rt, 0], [3, rt, 0], [4, rt, 0]])*a
        """
        self.position = np.array([[1, rt, 0],

                                  [1, 0, 0],


                                  [2.5, 0.5*rt, 0], [3, 0, 0],
                                  [2.5, -0.5*rt, 0],          [4.5, -0.5*rt, 0],


                                  [-0.5, 2.5*rt, 0], [0, 2*rt, 0],
                                  [-0.5, 1.5*rt, 0],          [1.5, 1.5*rt, 0],

                                  [3, 2*rt, 0],

                                  [3, rt, 0]])*a

    def generate_type8(self, a=1):
        R"""
        Introduction:
            (3,6,3,6) triangles and hexagons. kagome.
            a: the edge length of a single tile.
            n: n*n lattices to generate. 
        example:
            import workflow_analysis as wa
            import matplotlib.pyplot as plt
            at = wa.archimedean_tilings()
            at.generate_type11(2)
            points = at.generate_lattices([12,3])
            fig,ax = plt.subplots()
            ax.scatter(points[:,0],points[:,1])
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
        source:
            def generate_honeycomb(self,a,n):
            uc = hoomd.lattice.unitcell(N=4,
                                    a1=[3*a, 0, 0],
                                    a2=[0, math.sqrt(3)*a, 0],
                                    a3=[0, 0, 1],
                                    dimensions=2,
                                    position=[[0,0,0], [1/2*a, math.sqrt(3)/2*a, 0],[3/2*a,math.sqrt(3)/2*a,0],[2*a,0,0]],
                                    type_name=['A', 'A','A','A'],
                                    diameter=[1,1,1,1]);
            self.system = hoomd.init.create_lattice(unitcell=uc, n=n);
        """
        rt = RT
        # N=6
        self.a1 = np.array([2, 0, 0])*a
        self.a2 = np.array([0, 2*rt, 0])*a
        self.a3 = np.array([0, 0, 0])
        # dimensions=2
        self.position = np.array([[1.5, 1.5*rt, 0],
                                  [0, rt, 0], [1, rt, 0],
                                  [0, 0, 0], [1, 0, 0], [0.5, 0.5*rt, 0]])*a

    def generate_type8_part(self, a=1):
        R"""
        Introduction:
            (3,6,3,6) triangles and hexagons(kagome). 
            a: the edge length of a single tile.
            n: n*n lattices to generate. 
        example:
            import workflow_analysis as wa
            import matplotlib.pyplot as plt
            at = wa.archimedean_tilings()
            at.generate_type11(2)
            points = at.generate_lattices([12,3])
            fig,ax = plt.subplots()
            ax.scatter(points[:,0],points[:,1])
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
        source:
            def generate_honeycomb(self,a,n):
            uc = hoomd.lattice.unitcell(N=4,
                                    a1=[3*a, 0, 0],
                                    a2=[0, math.sqrt(3)*a, 0],
                                    a3=[0, 0, 1],
                                    dimensions=2,
                                    position=[[0,0,0], [1/2*a, math.sqrt(3)/2*a, 0],[3/2*a,math.sqrt(3)/2*a,0],[2*a,0,0]],
                                    type_name=['A', 'A','A','A'],
                                    diameter=[1,1,1,1]);
            self.system = hoomd.init.create_lattice(unitcell=uc, n=n);
        """
        rt = RT
        # N=4
        self.a1 = np.array([2, 0, 0])*a
        self.a2 = np.array([0, 2*rt, 0])*a
        self.a3 = np.array([0, 0, 0])
        # dimensions=2
        self.position = np.array([[0, rt, 0], [1, rt, 0],
                                  [0, 0, 0], [1, 0, 0]])*a

    def generate_type7(self, a=1):
        R"""
        Introduction:
            (3,4,6,4) triangles, squares and hexagons.
            a: the edge length of a single tile.
            n: n*n lattices to generate. 
        example:
            import workflow_analysis as wa
            import matplotlib.pyplot as plt
            at = wa.archimedean_tilings()
            at.generate_type11(2)
            points = at.generate_lattices([12,3])
            fig,ax = plt.subplots()
            ax.scatter(points[:,0],points[:,1])
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
        source:
            def generate_honeycomb(self,a,n):
            uc = hoomd.lattice.unitcell(N=4,
                                    a1=[3*a, 0, 0],
                                    a2=[0, math.sqrt(3)*a, 0],
                                    a3=[0, 0, 1],
                                    dimensions=2,
                                    position=[[0,0,0], [1/2*a, math.sqrt(3)/2*a, 0],[3/2*a,math.sqrt(3)/2*a,0],[2*a,0,0]],
                                    type_name=['A', 'A','A','A'],
                                    diameter=[1,1,1,1]);
            self.system = hoomd.init.create_lattice(unitcell=uc, n=n);
        """
        rt = RT
        # N=6
        self.a1 = np.array([1+rt, 0, 0])*a
        self.a2 = np.array([-0.5-0.5*rt, 1.5+0.5*rt, 0])*a  # (rt+3)*0.5
        self.a3 = np.array([0, 0, 0])
        # dimensions=2
        self.position = np.array([[-0.5, 1+0.5*rt, 0], [0.5*rt, 1.5, 0],
                                  [0, 1, 0], [rt, 1, 0],
                                  [0, 0, 0], [rt, 0, 0]])*a

    def generate_type7_part(self, a=1):
        R"""
        Introduction:
            (3,4,6,4) triangles, squares and hexagons.
            a: the edge length of a single tile.
            n: n*n lattices to generate. 
        example:
            import workflow_analysis as wa
            import matplotlib.pyplot as plt
            at = wa.archimedean_tilings()
            at.generate_type11(2)
            points = at.generate_lattices([12,3])
            fig,ax = plt.subplots()
            ax.scatter(points[:,0],points[:,1])
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
        source:
            def generate_honeycomb(self,a,n):
            uc = hoomd.lattice.unitcell(N=4,
                                    a1=[3*a, 0, 0],
                                    a2=[0, math.sqrt(3)*a, 0],
                                    a3=[0, 0, 1],
                                    dimensions=2,
                                    position=[[0,0,0], [1/2*a, math.sqrt(3)/2*a, 0],[3/2*a,math.sqrt(3)/2*a,0],[2*a,0,0]],
                                    type_name=['A', 'A','A','A'],
                                    diameter=[1,1,1,1]);
            self.system = hoomd.init.create_lattice(unitcell=uc, n=n);
        """
        rt = RT
        # N=5
        self.a1 = np.array([1+rt, 0, 0])*a
        self.a2 = np.array([-0.5-0.5*rt, 1.5+0.5*rt, 0])*a  # (rt+3)*0.5
        self.a3 = np.array([0, 0, 0])
        # dimensions=2
        self.position = np.array([[-0.5, 1+0.5*rt, 0],
                                  [0, 1, 0], [rt, 1, 0],
                                  [0, 0, 0], [rt, 0, 0]])*a

    def generate_type7_superlattice(self, a=1):
        R"""
        Introduction:
            (3,4,6,4) triangles, squares and hexagons.
            a: the edge length of a single tile.
            n: n*n lattices to generate. 
        example:
            import workflow_analysis as wa
            import matplotlib.pyplot as plt
            at = wa.archimedean_tilings()
            at.generate_type11(2)
            points = at.generate_lattices([12,3])
            fig,ax = plt.subplots()
            ax.scatter(points[:,0],points[:,1])
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
        source:
            def generate_honeycomb(self,a,n):
            uc = hoomd.lattice.unitcell(N=4,
                                    a1=[3*a, 0, 0],
                                    a2=[0, math.sqrt(3)*a, 0],
                                    a3=[0, 0, 1],
                                    dimensions=2,
                                    position=[[0,0,0], [1/2*a, math.sqrt(3)/2*a, 0],[3/2*a,math.sqrt(3)/2*a,0],[2*a,0,0]],
                                    type_name=['A', 'A','A','A'],
                                    diameter=[1,1,1,1]);
            self.system = hoomd.init.create_lattice(unitcell=uc, n=n);
        """
        rt = RT
        # N=6*4
        self.a1 = np.array([2+2*rt, 0, 0])*a  # 1+rt
        # [-0.5-0.5*rt,  1.5+0.5*rt, [0.5+0.5*rt,  1.5+0.5*rt]
        self.a2 = np.array([-1-rt, 3+rt, 0])*a
        self.a3 = np.array([0, 0, 0])
        # dimensions=2
        self.position = np.array(
            [[-1 - 0.5 * rt, 2.5 + rt, 0],
             [-0.5, 3 + 0.5 * rt, 0],
             [0.5 * rt, 2.5 + rt, 0],
             [0.5 + rt, 3 + 0.5 * rt, 0],
             [-0.5 - 0.5 * rt, 2.5 + 0.5 * rt, 0],
             [-0.5 + 0.5 * rt, 2.5 + 0.5 * rt, 0],
             [0.5 + 0.5 * rt, 2.5 + 0.5 * rt, 0],
             [0.5 + 1.5 * rt, 2.5 + 0.5 * rt, 0],
             [-0.5 - 0.5 * rt, 1.5 + 0.5 * rt, 0],
             [-0.5 + 0.5 * rt, 1.5 + 0.5 * rt, 0],
             [0.5 + 0.5 * rt, 1.5 + 0.5 * rt, 0],
             [0.5 + 1.5 * rt, 1.5 + 0.5 * rt, 0],
             [-0.5, 1 + 0.5 * rt, 0],
             [0.5 * rt, 1.5, 0],
             [0.5 + rt, 1 + 0.5 * rt, 0],
             [1 + 1.5 * rt, 1.5, 0],
             [0, 1, 0],
             [rt, 1, 0],
             [1 + rt, 1, 0],
             [1 + 2 * rt, 1, 0],
             [0, 0, 0],
             [rt, 0, 0],
             [1 + rt, 0, 0],
             [1 + 2 * rt, 0, 0]]) * a

    def generate_type7_part_superlattice(self, a=1):
        R"""
        Introduction:
            (3,4,6,4) triangles, squares and hexagons.
            a: the edge length of a single tile.
            n: n*n lattices to generate. 
        example:
            import workflow_analysis as wa
            import matplotlib.pyplot as plt
            at = wa.archimedean_tilings()
            at.generate_type11(2)
            points = at.generate_lattices([12,3])
            fig,ax = plt.subplots()
            ax.scatter(points[:,0],points[:,1])
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
        source:
            def generate_honeycomb(self,a,n):
            uc = hoomd.lattice.unitcell(N=4,
                                    a1=[3*a, 0, 0],
                                    a2=[0, math.sqrt(3)*a, 0],
                                    a3=[0, 0, 1],
                                    dimensions=2,
                                    position=[[0,0,0], [1/2*a, math.sqrt(3)/2*a, 0],[3/2*a,math.sqrt(3)/2*a,0],[2*a,0,0]],
                                    type_name=['A', 'A','A','A'],
                                    diameter=[1,1,1,1]);
            self.system = hoomd.init.create_lattice(unitcell=uc, n=n);
        """
        rt = RT
        # N=12
        self.a1 = np.array([2+2*rt, 0, 0])*a  # 1+rt
        # [-0.5-0.5*rt,  1.5+0.5*rt, [0.5+0.5*rt,  1.5+0.5*rt]
        self.a2 = np.array([-1-rt, 3+rt, 0])*a
        self.a3 = np.array([0, 0, 0])
        # dimensions=2
        """
        self.position = np.array([[-1-0.5*rt,2.5+rt,0],[-0.5,3+0.5*rt,0],                   [0.5*rt,2.5+rt,0],[0.5+rt,3+0.5*rt,0],
                                  [-0.5-0.5*rt,2.5+0.5*rt,0],[-0.5+0.5*rt,2.5+0.5*rt,0],    [0.5+0.5*rt,2.5+0.5*rt,0],[0.5+1.5*rt,2.5+0.5*rt,0],
                                  [-0.5-0.5*rt,1.5+0.5*rt,0],[-0.5+0.5*rt,1.5+0.5*rt,0],    [0.5+0.5*rt,1.5+0.5*rt,0],[0.5+1.5*rt,1.5+0.5*rt,0],

                                  [-0.5,1+0.5*rt,0],[0.5*rt,1.5,0], [0.5+rt,1+0.5*rt,0],[1+1.5*rt,1.5,0],
                                  [0,1,0],[rt,1,0],                 [1+rt,1,0],[1+2*rt,1,0],
                                  [0,0,0],[rt,0,0],                 [1+rt,0,0],[1+2*rt,0,0]
                                              ])*a
        """
        self.position = np.array([[-0.5, 3+0.5*rt, 0],                   [0.5*rt, 2.5+rt, 0],
                                  [0.5+0.5*rt, 2.5+0.5*rt, 0], [0.5+1.5*rt, 2.5+0.5*rt, 0],
                                  [-0.5-0.5*rt, 1.5+0.5*rt, 0], [-0.5+0.5*rt, 1.5+0.5*rt, 0],

                                  [0.5*rt, 1.5, 0], [0.5+rt, 1+0.5*rt, 0],
                                  [1+rt, 1, 0], [1+2*rt, 1, 0],
                                  [0, 0, 0], [rt, 0, 0],])*a

    def generate_type6(self, a=1):
        R"""
        Introduction:
            (4,8^2) squares and octagons.
            a: the edge length of a single tile.
            n: n*n lattices to generate. 
        example:
            import workflow_analysis as wa
            import matplotlib.pyplot as plt
            at = wa.archimedean_tilings()
            at.generate_type11(2)
            points = at.generate_lattices([12,3])
            fig,ax = plt.subplots()
            ax.scatter(points[:,0],points[:,1])
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
        source:
            def generate_honeycomb(self,a,n):
            uc = hoomd.lattice.unitcell(N=4,
                                    a1=[3*a, 0, 0],
                                    a2=[0, math.sqrt(3)*a, 0],
                                    a3=[0, 0, 1],
                                    dimensions=2,
                                    position=[[0,0,0], [1/2*a, math.sqrt(3)/2*a, 0],[3/2*a,math.sqrt(3)/2*a,0],[2*a,0,0]],
                                    type_name=['A', 'A','A','A'],
                                    diameter=[1,1,1,1]);
            self.system = hoomd.init.create_lattice(unitcell=uc, n=n);
        """
        rt = math.sqrt(2)
        # N=8
        self.a1 = np.array([1+rt, 0, 0])*a
        self.a2 = np.array([0, 1+rt, 0])*a
        self.a3 = np.array([0, 0, 0])
        # dimensions=2
        self.position = np.array([[0, 0.5*rt, 0], [0, 1+0.5*rt, 0],
                                  [0.5*rt, 0, 0], [1+0.5*rt, 0, 0]])*a

    def generate_type6_part(self, a=1):
        R"""
        Introduction:
            (4,8^2) squares and octagons.
            a: the edge length of a single tile.
            n: n*n lattices to generate. 
        example:
            import workflow_analysis as wa
            import matplotlib.pyplot as plt
            at = wa.archimedean_tilings()
            at.generate_type11(2)
            points = at.generate_lattices([12,3])
            fig,ax = plt.subplots()
            ax.scatter(points[:,0],points[:,1])
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
        source:
            def generate_honeycomb(self,a,n):
            uc = hoomd.lattice.unitcell(N=4,
                                    a1=[3*a, 0, 0],
                                    a2=[0, math.sqrt(3)*a, 0],
                                    a3=[0, 0, 1],
                                    dimensions=2,
                                    position=[[0,0,0], [1/2*a, math.sqrt(3)/2*a, 0],[3/2*a,math.sqrt(3)/2*a,0],[2*a,0,0]],
                                    type_name=['A', 'A','A','A'],
                                    diameter=[1,1,1,1]);
            self.system = hoomd.init.create_lattice(unitcell=uc, n=n);
        """
        rt = math.sqrt(2)
        # N=8
        self.a1 = np.array([1+rt, 0, 0])*a
        self.a2 = np.array([0, 1+rt, 0])*a
        self.a3 = np.array([0, 0, 0])
        # dimensions=2
        self.position = np.array([[0, 0.5*rt, 0], [0, 1+0.5*rt, 0],
                                  [0.5*rt, 0, 0]])*a

    def generate_type6_superlattice(self, a=1):
        R"""
        Introduction:
            (4,8^2) squares and octagons.
            a: the edge length of a single tile.
            n: n*n lattices to generate. 
        example:
            import workflow_analysis as wa
            import matplotlib.pyplot as plt
            at = wa.archimedean_tilings()
            at.generate_type11(2)
            points = at.generate_lattices([12,3])
            fig,ax = plt.subplots()
            ax.scatter(points[:,0],points[:,1])
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
        source:
            def generate_honeycomb(self,a,n):
            uc = hoomd.lattice.unitcell(N=4,
                                    a1=[3*a, 0, 0],
                                    a2=[0, math.sqrt(3)*a, 0],
                                    a3=[0, 0, 1],
                                    dimensions=2,
                                    position=[[0,0,0], [1/2*a, math.sqrt(3)/2*a, 0],[3/2*a,math.sqrt(3)/2*a,0],[2*a,0,0]],
                                    type_name=['A', 'A','A','A'],
                                    diameter=[1,1,1,1]);
            self.system = hoomd.init.create_lattice(unitcell=uc, n=n);
        """
        rt = math.sqrt(2)
        # N=8
        self.a1 = np.array([2+2*rt, 0, 0])*a
        self.a2 = np.array([0, 2+2*rt, 0])*a
        self.a3 = np.array([0, 0, 0])
        # dimensions=2
        # raw 2*2 superlattice
        self.position = np.array(
            [[0, 1 + 1.5 * rt, 0],
             [0, 2 + 1.5 * rt, 0],
             [1 + rt, 1 + 1.5 * rt, 0],
             [1 + rt, 2 + 1.5 * rt, 0],
             [0.5 * rt, 1 + rt, 0],
             [1 + 0.5 * rt, 1 + rt, 0],
             [1 + 1.5 * rt, 1 + rt, 0],
             [2 + 1.5 * rt, 1 + rt, 0],
             [0, 0.5 * rt, 0],
             [0, 1 + 0.5 * rt, 0],
             [1 + rt, 0.5 * rt, 0],
             [1 + rt, 1 + 0.5 * rt, 0],
             [0.5 * rt, 0, 0],
             [1 + 0.5 * rt, 0, 0],
             [1 + 1.5 * rt, 0, 0],
             [2 + 1.5 * rt, 0, 0]]) * a

    def generate_type6_part_superlattice(self, a=1):
        R"""
        Introduction:
            (4,8^2) squares and octagons.
            a: the edge length of a single tile.
            n: n*n lattices to generate. 
        example:
            import workflow_analysis as wa
            import matplotlib.pyplot as plt
            at = wa.archimedean_tilings()
            at.generate_type11(2)
            points = at.generate_lattices([12,3])
            fig,ax = plt.subplots()
            ax.scatter(points[:,0],points[:,1])
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
        source:
            def generate_honeycomb(self,a,n):
            uc = hoomd.lattice.unitcell(N=4,
                                    a1=[3*a, 0, 0],
                                    a2=[0, math.sqrt(3)*a, 0],
                                    a3=[0, 0, 1],
                                    dimensions=2,
                                    position=[[0,0,0], [1/2*a, math.sqrt(3)/2*a, 0],[3/2*a,math.sqrt(3)/2*a,0],[2*a,0,0]],
                                    type_name=['A', 'A','A','A'],
                                    diameter=[1,1,1,1]);
            self.system = hoomd.init.create_lattice(unitcell=uc, n=n);
        """
        rt = math.sqrt(2)
        # N=8
        self.a1 = np.array([2+2*rt, 0, 0])*a
        self.a2 = np.array([0, 2+2*rt, 0])*a
        self.a3 = np.array([0, 0, 0])
        # dimensions=2
        """#raw 2*2 superlattice
        self.position = np.array([[0,1+1.5*rt,0],[0,2+1.5*rt,0],      [1+rt,1+1.5*rt,0],[1+rt,2+1.5*rt,0],
                                  [0.5*rt,1+rt,0],[1+0.5*rt,1+rt,0],  [1+1.5*rt,1+rt,0],[2+1.5*rt,1+rt,0],
                                  [0,0.5*rt,0],[0,1+0.5*rt,0],        [1+rt,0.5*rt,0],[1+rt,1+0.5*rt,0],
                                  [0.5*rt,0,0],[1+0.5*rt,0,0],        [1+1.5*rt,0,0],[2+1.5*rt,0,0] ])*a
        """
        self.position = np.array(
            [[0, 1 + 1.5 * rt, 0],
             [1 + rt, 2 + 1.5 * rt, 0],
             [1 + 0.5 * rt, 1 + rt, 0],
             [1 + 1.5 * rt, 1 + rt, 0],
             [0, 1 + 0.5 * rt, 0],
             [1 + rt, 0.5 * rt, 0],
             [0.5 * rt, 0, 0],
             [2 + 1.5 * rt, 0, 0]]) * a

    def generate_type5(self, a=1):
        R"""
        Introduction:
            (4,6,12) squares, hexagons and dodecagons.
            a: the edge length of a single tile.
            n: n*n lattices to generate. 
        example:
            import workflow_analysis as wa
            import matplotlib.pyplot as plt
            at = wa.archimedean_tilings()
            at.generate_type11(2)
            points = at.generate_lattices([12,3])
            fig,ax = plt.subplots()
            ax.scatter(points[:,0],points[:,1])
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
        source:
            def generate_honeycomb(self,a,n):
            uc = hoomd.lattice.unitcell(N=4,
                                    a1=[3*a, 0, 0],
                                    a2=[0, math.sqrt(3)*a, 0],
                                    a3=[0, 0, 1],
                                    dimensions=2,
                                    position=[[0,0,0], [1/2*a, math.sqrt(3)/2*a, 0],[3/2*a,math.sqrt(3)/2*a,0],[2*a,0,0]],
                                    type_name=['A', 'A','A','A'],
                                    diameter=[1,1,1,1]);
            self.system = hoomd.init.create_lattice(unitcell=uc, n=n);
        """
        rt = RT
        # N=12
        self.a1 = np.array([3+rt, 0, 0])*a
        self.a2 = np.array([-1.5-0.5*rt, 1.5+1.5*rt, 0])*a  # (rt+3)*0.5
        self.a3 = np.array([0, 0, 0])
        # dimensions=2
        self.position = np.array([[-1.5, 1+1.5*rt, 0], [-0.5, 1+1.5*rt, 0],
                                  [0.5*rt, 1.5+rt, 0], [1+0.5*rt, 1.5+rt, 0],
                                  [0, 1+rt, 0], [1+rt, 1+rt, 0],
                                  [-0.5, 1+0.5*rt, 0], [1.5+rt, 1+0.5*rt, 0],
                                  [-0.5, 0.5*rt, 0], [1.5+rt, 0.5*rt, 0],
                                  [0, 0, 0], [1+rt, 0, 0]])*a

    def generate_type5_part(self, a=1):
        R"""
        Introduction:
            (4,6,12) squares, hexagons and dodecagons.
            a: the edge length of a single tile.
            n: n*n lattices to generate. 
        example:
            import workflow_analysis as wa
            import matplotlib.pyplot as plt
            at = wa.archimedean_tilings()
            at.generate_type11(2)
            points = at.generate_lattices([12,3])
            fig,ax = plt.subplots()
            ax.scatter(points[:,0],points[:,1])
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
        source:
            def generate_honeycomb(self,a,n):
            uc = hoomd.lattice.unitcell(N=4,
                                    a1=[3*a, 0, 0],
                                    a2=[0, math.sqrt(3)*a, 0],
                                    a3=[0, 0, 1],
                                    dimensions=2,
                                    position=[[0,0,0], [1/2*a, math.sqrt(3)/2*a, 0],[3/2*a,math.sqrt(3)/2*a,0],[2*a,0,0]],
                                    type_name=['A', 'A','A','A'],
                                    diameter=[1,1,1,1]);
            self.system = hoomd.init.create_lattice(unitcell=uc, n=n);
        """
        rt = RT
        # N=11
        self.a1 = np.array([3+rt, 0, 0])*a
        self.a2 = np.array([-1.5-0.5*rt, 1.5+1.5*rt, 0])*a  # (rt+3)*0.5
        self.a3 = np.array([0, 0, 0])
        # dimensions=2
        self.position = np.array([[-1.5, 1+1.5*rt, 0],
                                  [0.5*rt, 1.5+rt, 0], [1+0.5*rt, 1.5+rt, 0],
                                  [0, 1+rt, 0], [1+rt, 1+rt, 0],
                                  [-0.5, 1+0.5*rt, 0], [1.5+rt, 1+0.5*rt, 0],
                                  [-0.5, 0.5*rt, 0], [1.5+rt, 0.5*rt, 0],
                                  [0, 0, 0], [1+rt, 0, 0]])*a

    def generate_type4(self, a=1):
        R"""
        Introduction:
            (3,12^2) triangles and dodecagons.
            a: the edge length of a single tile.
            n: n*n lattices to generate. 
        example:
            import workflow_analysis as wa
            import matplotlib.pyplot as plt
            at = wa.archimedean_tilings()
            at.generate_type11(2)
            points = at.generate_lattices([12,3])
            fig,ax = plt.subplots()
            ax.scatter(points[:,0],points[:,1])
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
        source:
            def generate_honeycomb(self,a,n):
            uc = hoomd.lattice.unitcell(N=4,
                                    a1=[3*a, 0, 0],
                                    a2=[0, math.sqrt(3)*a, 0],
                                    a3=[0, 0, 1],
                                    dimensions=2,
                                    position=[[0,0,0], [1/2*a, math.sqrt(3)/2*a, 0],[3/2*a,math.sqrt(3)/2*a,0],[2*a,0,0]],
                                    type_name=['A', 'A','A','A'],
                                    diameter=[1,1,1,1]);
            self.system = hoomd.init.create_lattice(unitcell=uc, n=n);
        """
        rt = RT
        # N=6
        self.a1 = np.array([2+rt, 0, 0])*a
        self.a2 = np.array([-1-0.5*rt, 1.5+rt, 0])*a  # (rt+3)*0.5
        self.a3 = np.array([0, 0, 0])
        # dimensions=2
        self.position = np.array([[-1, 1+rt, 0], [0, 1+rt, 0],
                                  [-0.5, 1+0.5*rt, 0],
                                  [-0.5, 0.5*rt, 0],
                                  [0, 0, 0], [1+rt, 0, 0]])*a

    def generate_type4_part(self, a=1):
        R"""
        Introduction:
            (3,12^2) triangles and dodecagons.
            a: the edge length of a single tile.
            n: n*n lattices to generate. 
        example:
            import workflow_analysis as wa
            import matplotlib.pyplot as plt
            at = wa.archimedean_tilings()
            at.generate_type11(2)
            points = at.generate_lattices([12,3])
            fig,ax = plt.subplots()
            ax.scatter(points[:,0],points[:,1])
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
        source:
            def generate_honeycomb(self,a,n):
            uc = hoomd.lattice.unitcell(N=4,
                                    a1=[3*a, 0, 0],
                                    a2=[0, math.sqrt(3)*a, 0],
                                    a3=[0, 0, 1],
                                    dimensions=2,
                                    position=[[0,0,0], [1/2*a, math.sqrt(3)/2*a, 0],[3/2*a,math.sqrt(3)/2*a,0],[2*a,0,0]],
                                    type_name=['A', 'A','A','A'],
                                    diameter=[1,1,1,1]);
            self.system = hoomd.init.create_lattice(unitcell=uc, n=n);
        """
        rt = RT
        # N=4
        self.a1 = np.array([2+rt, 0, 0])*a
        self.a2 = np.array([-1-0.5*rt, 1.5+rt, 0])*a  # (rt+3)*0.5
        self.a3 = np.array([0, 0, 0])
        # dimensions=2
        self.position = np.array([[-1, 1+rt, 0],
                                  [-0.5, 1+0.5*rt, 0],
                                  [0, 0, 0], [1+rt, 0, 0]])*a

    def generate_type3(self, a=1):
        R"""
        Introduction:
            (6^3) hexagons(honeycomb).
            a: the edge length of a single tile.
        example:
            import workflow_analysis as wa
            import matplotlib.pyplot as plt
            at = wa.archimedean_tilings()
            at.generate_type11(2)
            points = at.generate_lattices([12,3])
            fig,ax = plt.subplots()
            ax.scatter(points[:,0],points[:,1])
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
        source:
            def generate_honeycomb(self,a,n):
            uc = hoomd.lattice.unitcell(N=4,
                                    a1=[3*a, 0, 0],
                                    a2=[0, math.sqrt(3)*a, 0],
                                    a3=[0, 0, 1],
                                    dimensions=2,
                                    position=[[0,0,0], [1/2*a, math.sqrt(3)/2*a, 0],[3/2*a,math.sqrt(3)/2*a,0],[2*a,0,0]],
                                    type_name=['A', 'A','A','A'],
                                    diameter=[1,1,1,1]);
            self.system = hoomd.init.create_lattice(unitcell=uc, n=n);
        """
        rt = RT
        # N=4
        self.a1 = np.array([3, 0, 0])*a
        self.a2 = np.array([0, rt, 0])*a
        self.a3 = np.array([0, 0, 0])
        # dimensions=2
        self.position = np.array([[0, 0, 0], [1/2, rt/2, 0], [3/2, rt/2, 0], [2, 0, 0]])*a

    def generate_type3_part(self, a=1):
        R"""
        Introduction:
            (3^6) triangles(honeycomb_part).
            a: the edge length of a single tile.
        example:
            import workflow_analysis as wa
            import matplotlib.pyplot as plt
            at = wa.archimedean_tilings()
            at.generate_type11(2)
            points = at.generate_lattices([12,3])
            fig,ax = plt.subplots()
            ax.scatter(points[:,0],points[:,1])
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
        source:
            def generate_honeycomb(self,a,n):
            uc = hoomd.lattice.unitcell(N=4,
                                    a1=[3*a, 0, 0],
                                    a2=[0, math.sqrt(3)*a, 0],
                                    a3=[0, 0, 1],
                                    dimensions=2,
                                    position=[[0,0,0], [1/2*a, math.sqrt(3)/2*a, 0],[3/2*a,math.sqrt(3)/2*a,0],[2*a,0,0]],
                                    type_name=['A', 'A','A','A'],
                                    diameter=[1,1,1,1]);
            self.system = hoomd.init.create_lattice(unitcell=uc, n=n);
        """
        rt = RT
        # N=2
        self.a1 = np.array([3, 0, 0])*a
        self.a2 = np.array([0, rt, 0])*a
        self.a3 = np.array([0, 0, 0])
        # dimensions=2
        self.position = np.array([[0, 0, 0], [3/2, rt/2, 0]])*a

    def generate_type2(self, a=1):
        R"""
        Introduction:
            (4^4) squares.
            a: the edge length of a single tile.
        example:
            import workflow_analysis as wa
            import matplotlib.pyplot as plt
            at = wa.archimedean_tilings()
            at.generate_type11(2)
            points = at.generate_lattices([12,3])
            fig,ax = plt.subplots()
            ax.scatter(points[:,0],points[:,1])
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
        source:
            def generate_honeycomb(self,a,n):
            uc = hoomd.lattice.unitcell(N=4,
                                    a1=[3*a, 0, 0],
                                    a2=[0, math.sqrt(3)*a, 0],
                                    a3=[0, 0, 1],
                                    dimensions=2,
                                    position=[[0,0,0], [1/2*a, math.sqrt(3)/2*a, 0],[3/2*a,math.sqrt(3)/2*a,0],[2*a,0,0]],
                                    type_name=['A', 'A','A','A'],
                                    diameter=[1,1,1,1]);
            self.system = hoomd.init.create_lattice(unitcell=uc, n=n);
        """
        # N=1
        self.a1 = np.array([1, 0, 0])*a
        self.a2 = np.array([0, 1, 0])*a
        self.a3 = np.array([0, 0, 0])
        # dimensions=2
        self.position = np.array([[0, 0, 0]])*a

    def generate_type1(self, a=1):
        R"""
        Introduction:
            (3^6) triangles(hexagonal).
            a: the edge length of a single tile.
        example:
            import workflow_analysis as wa
            import matplotlib.pyplot as plt
            at = wa.archimedean_tilings()
            at.generate_type11(2)
            points = at.generate_lattices([12,3])
            fig,ax = plt.subplots()
            ax.scatter(points[:,0],points[:,1])
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
        source:
            def generate_honeycomb(self,a,n):
            uc = hoomd.lattice.unitcell(N=4,
                                    a1=[3*a, 0, 0],
                                    a2=[0, math.sqrt(3)*a, 0],
                                    a3=[0, 0, 1],
                                    dimensions=2,
                                    position=[[0,0,0], [1/2*a, math.sqrt(3)/2*a, 0],[3/2*a,math.sqrt(3)/2*a,0],[2*a,0,0]],
                                    type_name=['A', 'A','A','A'],
                                    diameter=[1,1,1,1]);
            self.system = hoomd.init.create_lattice(unitcell=uc, n=n);
        """
        rt = RT
        # N=2
        self.a1 = np.array([1, 0, 0])*a
        self.a2 = np.array([0, rt, 0])*a
        self.a3 = np.array([0, 0, 0])
        # dimensions=2
        self.position = np.array([[0, 0, 0], [0.5, 0.5*rt, 0]])*a

    def generate_type_n(self, type_n, a=1):
        if (type_n) == 1:
            self.generate_type1(a)
        elif (type_n) == 2:
            self.generate_type2(a)
        elif (type_n) == 3:
            self.generate_type3(a)
        elif (type_n) == 4:
            self.generate_type4(a)
        elif (type_n) == 5:
            self.generate_type5(a)
        elif (type_n) == 6:
            self.generate_type6(a)
        elif (type_n) == 7:
            self.generate_type7(a)
        elif (type_n) == 8:
            self.generate_type8(a)
        elif (type_n) == 9:
            self.generate_type9(a)
        elif (type_n) == 10:
            self.generate_type10(a)
        elif (type_n) == 11:
            self.generate_type11(a)
        elif (type_n) == 62:
            self.generate_type6_superlattice(a)
        elif (type_n) == 72:
            self.generate_type7_superlattice(a)
        elif (type_n) == 92:
            self.generate_type9_superlattice(a)

    def generate_type_n_part(self, type_n, a=1):
        if (type_n) == 1:
            self.generate_type1(a)
        elif (type_n) == 2:
            self.generate_type2(a)
        elif (type_n) == 3:
            self.generate_type3_part(a)
        elif (type_n) == 4:
            self.generate_type4_part(a)
        elif (type_n) == 5:
            self.generate_type5_part(a)
        elif (type_n) == 6:
            self.generate_type6_part(a)
        elif (type_n) == 7:
            self.generate_type7_part(a)
        elif (type_n) == 8:
            self.generate_type8_part(a)
        elif (type_n) == 9:
            self.generate_type9_part(a)
        elif (type_n) == 10:
            self.generate_type10_part(a)
        elif (type_n) == 11:
            self.generate_type11_part(a)
        elif (type_n) == 62:
            self.generate_type6_part_superlattice(a)
        elif (type_n) == 72:
            self.generate_type7_part_superlattice(a)
        elif (type_n) == 92:
            self.generate_type9_part_superlattice(a)

    def generate_lattices(self, n):
        R"""
            self.position: an array of points as a lattice to generate a larger crystal.
            n: the size of lattice to expand. 
                n=5 to generate 5*5 lattice; 
                n=[5,10] to generate  5*10 lattice.
        """
        sz = np.shape(self.position)
        # ll = len(n)
        if isinstance(n, int):
            # positions = np.zeros((sz[0]*n*n,sz[1]))
            positions = self.generate_lattices([n, n])
        elif len(n) == int(2):
            positions = np.zeros((sz[0]*n[0]*n[1], sz[1]))
            for i in range(n[0]):
                position_temp = self.position + self.a1*i
                positions[i*sz[0]:(i+1)*sz[0], :] = position_temp

            for i in range(n[1]):
                if i > 0:
                    position_temp = positions[:sz[0]*n[0], :] + self.a2*i
                    positions[i*sz[0]*n[0]:(i+1)*sz[0]*n[0], :] = position_temp
            # centralize the positions.
            central = self.a1/2.0*n[0] + self.a2/2.0*n[1] + self.a3/2.0
            positions[:] = positions[:] - central
        else:
            print('Error: n must be a positive int num or 2d int array!')

        return positions

    def generate_lattices_not_centralized(self, n):
        R"""
            self.position: an array of points as a lattice to generate a larger crystal.
            n: the size of lattice to expand. 
                n=5 to generate 5*5 lattice; 
                n=[5,10] to generate  5*10 lattice.
        """
        sz = np.shape(self.position)
        # ll = len(n)
        if isinstance(n, int):
            # positions = np.zeros((sz[0]*n*n,sz[1]))
            positions = self.generate_lattices_not_centralized([n, n])
        elif len(n) == int(2):
            positions = np.zeros((sz[0]*n[0]*n[1], sz[1]))
            for i in range(n[0]):
                position_temp = self.position + self.a1*i
                positions[i*sz[0]:(i+1)*sz[0], :] = position_temp

            for i in range(n[1]):
                if i > 0:
                    position_temp = positions[:sz[0]*n[0], :] + self.a2*i
                    positions[i*sz[0]*n[0]:(i+1)*sz[0]*n[0], :] = position_temp
        else:
            print('Error: n must be a positive int num or 2d int array!')

        return positions

    def get_dual_subtract_type_n_part_complement(self, a, pos_all, pos_part):
        R"""
        Introduction:
            calculate the distance between dual lattice of type_n_part and type_n_part_complement(dual_p1).
            select the points whose distance is larger than 0.1*a, named as dual_type_n_part_interstitial(dual_p2).
        """
        dual_subtract = self.get_positions_in_all_but_not_in_part(a, pos_all, pos_part)
        return dual_subtract

    def get_positions_in_all_but_not_in_part(self, a, pos_all, pos_part):
        R"""
        Introduction:
            calculate the distance between pos_all and pos_part, where a is the typical distance between positions.
            select the points whose distance is larger than 0.1*a, named as pos_not_in_part.
        """
        pair_length = distance.cdist(pos_all, pos_part, 'euclidean')
        pair_length[pair_length < 0.1*a] = 0
        pair_length[pair_length > 0.1*a] = 1
        pair_large_length_bool = np.min(pair_length, axis=1)  # [:,0]
        pos_not_in_part = pos_all[pair_large_length_bool == 1]
        return pos_not_in_part

    def get_dual_lattice(self, points):
        test = pa.static_points_analysis_2d(points, hide_figure=False)
        pt = test.voronoi.vertices
        return pt

    def get_type_n_lcr0(self):
        R"""
        introduction:
            lcr0 is a parameter when a_hex * lcr0 = a_type_n, 
            the particle density of n_hex and n_type_n are equal. 
        return:
            record_lcr0:(11,)[lcr0_for_type1,2,3...,11]
        """
        record_lcr0 = np.zeros((11,))
        for i in range(11):
            self.generate_type_n(i+1)
            cross_lattice = np.cross(self.a1, self.a2)
            area_per_particle = cross_lattice[2]/len(self.position)
            area_hex = np.sqrt(3)/2.0
            lcr0 = np.sqrt(area_hex/area_per_particle)
            record_lcr0[i] = lcr0
            # print("type"+str(i+1)+": "+str(np.round(lcr0,4) ))
            # del at
        return record_lcr0

    def get_coordination_number_k_for_type_n(self, type_n):
        R"""
        inform the users using which order parameter 
        to evaluate the ratio of type_n transformation 
        """
        if type_n == 1:
            coord_num_k = 6
        elif type_n == 2:
            coord_num_k = 4
        elif type_n == 3:
            coord_num_k = 3
        elif type_n == 4:
            coord_num_k = 3
        elif type_n == 5:
            coord_num_k = 3
        elif type_n == 6:
            coord_num_k = 3
        elif type_n == 7:
            coord_num_k = 4
        elif type_n == 8:
            coord_num_k = 4
        elif type_n == 9:
            coord_num_k = 5
        elif type_n == 10:
            coord_num_k = 5
        elif type_n == 11:
            coord_num_k = 5
        elif type_n == 62:
            coord_num_k = 3
        elif type_n == 72:
            coord_num_k = 4

        return coord_num_k


class archimedean_tilings_polygon_dye:
    def __init__(self):
        R"""
        example:
            import workflow_analysis as wa
            at = wa.archimedean_tilings_polygon_dye()
            at.workflow_type4()
            at.workflow_type5()
            at.workflow_type6()
            at.workflow_type7()
            at.workflow_type9()
            at.workflow_type10()
            at.workflow_type11()
        """
        self.water_color = np.array([115, 163, 255])/255.0
        self.particle_color = 'k'

        # colorblind ibm-format
        self.color3 = np.array([255, 176, 0])/255.0
        self.color4 = np.array([254, 97, 0])/255.0
        self.color6 = np.array([220, 38, 127])/255.0
        self.color8 = np.array([120, 94, 240])/255.0
        self.color12 = np.array([100, 143, 255])/255.0
        """#color2
        self.color3 = 'royalblue'
        self.color4 = 'forestgreen'
        self.color6 = 'r'
        self.color8 = 'violet'
        self.color12 = 'darkorange'#'mediumpurple'"""

        """self.color3 = 'r'
        self.color4 = 'forestgreen'
        self.color6 = 'darkorange'
        self.color8 = 'royalblue'
        self.color12 = 'mediumpurple'"""

    def workflow_type_n(self, type_n, xylim=5, n_plus=3, part=False):
        R"""
        import workflow_analysis as wa
        atpd = wa.archimedean_tilings_polygon_dye()
        for i in range(9):
            atpd.workflow_type_n(i+3)
        """
        at_part = archimedean_tilings()
        at_part.generate_type_n_part(type_n)  # <delta>
        png_filename = 'polygon_dye_colorblind_type'+str(type_n)+'.png'  # <delta>_colorblind
        vec = at_part.a1+at_part.a2
        n1 = int(np.around(2*xylim/vec[0], 0)+n_plus)
        n2 = int(np.around(2*xylim/vec[1], 0)+n_plus)
        points = at_part.generate_lattices([n1, n2])  # 1.73:2
        # dula = at.get_dual_lattice(points)
        fig, ax = plt.subplots()
        ax.scatter(points[:, 0], points[:, 1], color='k', zorder=3)
        # ax.scatter(dula[:,0],dula[:,1],facecolors='white',edgecolors='k',zorder=3)
        # draw bonds selected
        at_full = archimedean_tilings()
        at_full.generate_type_n(type_n)  # <delta>
        pointsb = at_full.generate_lattices([n1, n2])
        perturbation = np.random.random(pointsb.shape)*0.01
        pointsb = pointsb + perturbation  # precisely equalled bond will let delaunay disfunction!
        p2d = pa.static_points_analysis_2d(pointsb, hide_figure=False)
        p2d.get_first_minima_bond_length_distribution(
            lattice_constant=1)  # png_filename='bond_hist.png'
        bpm = pa.bond_plot_module(fig, ax)
        bpm.restrict_axis_property_relative(hide_axis=True)
        list_bond_index = bpm.get_bonds_with_conditional_bond_length(
            p2d.bond_length, [0.8, p2d.bond_first_minima_left])
        bpm.draw_points_with_given_bonds(pointsb, list_bond_index, bond_color='k', particle_size=1)
        del at_full
        # bpm.draw_points_with_given_bonds(points,list_bond_index,bond_color='k',particle_color='r')#p2d.bond_length[:,:2].astype(int)
        # draw polygons selected
        count_polygon_relative = p2d.get_conditional_bonds_and_simplices_bond_length()
        fig, ax = p2d.draw_polygon_patch_oop(fig, ax, self.color3, polygon_n=3)
        fig, ax = p2d.draw_polygon_patch_oop(fig, ax, self.color4, polygon_n=4)
        fig, ax = p2d.draw_polygon_patch_oop(fig, ax, self.color6, polygon_n=6)
        fig, ax = p2d.draw_polygon_patch_oop(fig, ax, self.color8, polygon_n=8)
        fig, ax = p2d.draw_polygon_patch_oop(fig, ax, self.color12, polygon_n=12)
        # ax.set_aspect('equal','box')
        ax.set_xlim([-xylim, xylim])
        ax.set_ylim([-xylim, xylim])
        bpm.save_figure(png_filename)  # plt.savefig(png_filename)
        # plt.close('all')
        del at_part

    def workflow_type_n_complement(self, type_n, xylim=5, n_plus=3):
        R"""
        import workflow_analysis as wa
        atpd = wa.archimedean_tilings_polygon_dye()
        for i in range(9):
            atpd.workflow_type_n_complement(i+3)
        """
        at_part = archimedean_tilings()
        at_part.generate_type_n_part(type_n)  # <delta>
        png_filename = 'archimedean_bond_colorblind_type_'+str(type_n)+'.png'  # <delta>_colorblind
        vec = at_part.a1+at_part.a2
        n1 = int(np.around(2*xylim/vec[0], 0)+n_plus)
        n2 = int(np.around(2*xylim/vec[1], 0)+n_plus)
        points = at_part.generate_lattices([n1, n2])  # 1.73:2
        position_dula = at_part.get_dual_lattice(points)
        fig, ax = plt.subplots()
        ax.scatter(points[:, 0], points[:, 1], facecolors='white',
                   edgecolors='k', zorder=3)  # ,color='k'
        ax.scatter(position_dula[:, 0], position_dula[:, 1], color=self.color6, zorder=3)
        # draw bonds selected
        at_full = archimedean_tilings()
        at_full.generate_type_n(type_n)  # <delta>
        pointsb = at_full.generate_lattices([n1, n2])
        perturbation = np.random.random(pointsb.shape)*0.01
        pointsb = pointsb + perturbation  # precisely equalled bond will let delaunay disfunction!
        p2d = pa.static_points_analysis_2d(pointsb, hide_figure=False)
        p2d.get_first_minima_bond_length_distribution(
            lattice_constant=1)  # png_filename='bond_hist.png'
        bpm = pa.bond_plot_module(fig, ax)
        bpm.restrict_axis_property_relative(hide_axis=True)
        list_bond_index = bpm.get_bonds_with_conditional_bond_length(
            p2d.bond_length, [0.8, p2d.bond_first_minima_left])
        bpm.draw_points_with_given_bonds(pointsb, list_bond_index,
                                         bond_color='gray', particle_size=1)  # 'k'
        points_complement = at_part.get_dual_subtract_type_n_part_complement(1, pointsb, points)
        ax.scatter(points_complement[:, 0], points_complement[:, 1], color=self.color8, zorder=4)
        del at_full
        # bpm.draw_points_with_given_bonds(points,list_bond_index,bond_color='k',particle_color='r')#p2d.bond_length[:,:2].astype(int)
        # draw polygons selected
        """count_polygon_relative = p2d.get_conditional_bonds_and_simplices_bond_length()
        fig,ax = p2d.draw_polygon_patch_oop(fig,ax,self.color3,polygon_n=3)
        fig,ax = p2d.draw_polygon_patch_oop(fig,ax,self.color4,polygon_n=4)
        fig,ax = p2d.draw_polygon_patch_oop(fig,ax,self.color6,polygon_n=6)
        fig,ax = p2d.draw_polygon_patch_oop(fig,ax,self.color8,polygon_n=8)
        fig,ax = p2d.draw_polygon_patch_oop(fig,ax,self.color12,polygon_n=12)"""
        # ax.set_aspect('equal','box')
        ax.set_xlim([-xylim, xylim])
        ax.set_ylim([-xylim, xylim])
        bpm.save_figure(png_filename)  # plt.savefig(png_filename)
        # plt.close('all')
        del at_part

    def workflow_type_n_pure_background(
            self, type_n, xylim=5, n_plus=3, color_particle='k', part=False, follow_points=True):
        R"""
        import workflow_analysis as wa
        atpd = wa.archimedean_tilings_polygon_dye()
        for i in range(9):
            atpd.workflow_type_n(i+3)
        """
        particle_size = 300  # 300/xylim
        bond_width = 5  # 10/xylim
        at_part = archimedean_tilings()
        if part:
            at_part.generate_type_n_part(type_n)  # <delta>
            # <delta>_colorblind
            png_filename = 'polygon_dye_colorblind_type'+str(type_n)+'_part.png'
        else:
            at_part.generate_type_n(type_n)
            png_filename = 'polygon_dye_colorblind_type'+str(type_n)+'.png'  # <delta>_colorblind
        vec = at_part.a1+at_part.a2
        n1 = int(np.around(2*xylim/vec[0], 0)+n_plus)
        n2 = int(np.around(2*xylim/vec[1], 0)+n_plus)
        points = at_part.generate_lattices([n1, n2])  # 1.73:2
        # dula = at.get_dual_lattice(points)
        fig, ax = plt.subplots()
        ax.scatter(points[:, 0], points[:, 1], s=particle_size, color=color_particle, zorder=3)
        # ax.scatter(dula[:,0],dula[:,1],facecolors='white',edgecolors='k',zorder=3)
        # draw bonds selected
        if follow_points:
            pointsb = points
        else:
            at_full = archimedean_tilings()
            at_full.generate_type_n(type_n)  # <delta>
            pointsb = at_full.generate_lattices([n1, n2])
        perturbation = np.random.random(pointsb.shape)*0.01
        pointsb = pointsb + perturbation  # precisely equalled bond will let delaunay disfunction!
        p2d = pa.static_points_analysis_2d(pointsb, hide_figure=False)
        p2d.get_first_minima_bond_length_distribution(
            lattice_constant=1)  # png_filename='bond_hist.png'
        bpm = pa.bond_plot_module(fig, ax)
        bpm.restrict_axis_property_relative(hide_axis=True)
        # bpm.restrict_axis_property_relative(x_unit='(1)')
        list_bond_index = bpm.get_bonds_with_conditional_bond_length(
            p2d.bond_length, [0.8, p2d.bond_first_minima_left])
        bpm.draw_points_with_given_bonds(pointsb, list_bond_index,
                                         bond_color='k', bond_width=bond_width, particle_size=1)
        # del at_full
        # bpm.draw_points_with_given_bonds(points,list_bond_index,bond_color='k',particle_color='r')#p2d.bond_length[:,:2].astype(int)
        # draw polygons selected
        # bpm.plot_scale_bar(0, 0, 1)
        bpm.plot_background(0, 0, 10, self.color12), 'k'
        # ax.set_aspect('equal','box')
        ax.set_xlim([-xylim, xylim])
        ax.set_ylim([-xylim, xylim])
        bpm.save_figure(png_filename)  # plt.savefig(png_filename)
        # plt.close('all')
        del at_part


class show_dual_lattice:
    R"""
    import getDataAndScatter
    getDataAndScatter.get_dual_lattice()

    import workflow_analysis as wa
    sdl = wa.show_dual_lattice()
    sdl.show_dual_type11_part()

    sdl.show_dual_type10_part()
    sdl.show_dual_type9_part()
    sdl.show_dual_type8_part()
    sdl.show_dual_type7_part()
    sdl.show_dual_type6_part()
    sdl.show_dual_type5_part()
    sdl.show_dual_type4_part()
    """

    def __init__(self):
        pass

    def go(self):
        trap_filename = '/home/remote/hoomd-examples_0/testhoneycomb3-8-12'
        traps = np.loadtxt(trap_filename)
        LinearCompressionRatio = 1
        traps = np.multiply(traps, LinearCompressionRatio)
        fig, ax = plt.subplots()
        # ax.scatter(pos[:,0],pos[:,1],c='k')
        ax.scatter(traps[:, 0],
                   traps[:, 1],
                   c='r')  # ,marker = 'x'
        ax.set_aspect('equal', 'box')
        """
        limit=[[],[]]
        ax.set_xlim(limit[0])
        ax.set_ylim(limit[1])
        """

    def show_dual_type_n_part(self, type_n, xylim=5, bond_on=True, n_plus=2):
        R"""
        type_n: if 3, means the polygon is of type_3.(see archimedean_tilings)
        xylim: if 5, means the plot would be 10*10.

        examples:
            import workflow_analysis as wa
            sdl = wa.show_dual_lattice()
            for i in [3,6,7,8,9,10,11]:
                sdl.show_dual_type_n_part(type_n=i,xylim=3,n_plus=2)
        """
        at = archimedean_tilings()
        at.generate_type_n_part(type_n)  # <delta>
        png_filename = 'dual_type'+str(type_n)+'_part_bond.png'  # <delta>
        vec = at.a1+at.a2
        n1 = int(np.around(2*xylim/vec[0], 0)+n_plus)
        n2 = int(np.around(2*xylim/vec[1], 0)+n_plus)
        points = at.generate_lattices([n1, n2])  # 1.73:2
        dula = at.get_dual_lattice(points)
        fig, ax = plt.subplots()
        ax.scatter(points[:, 0], points[:, 1], color='k', zorder=3)
        ax.scatter(dula[:, 0], dula[:, 1], facecolors='white', edgecolors='k', zorder=3)
        # draw bonds selected
        if bond_on:
            atb = archimedean_tilings()
            atb.generate_type_n(type_n)  # <delta>
            pointsb = atb.generate_lattices([n1, n2])
            perturbation = np.random.random(pointsb.shape)*0.01
            pointsb = pointsb + perturbation  # precisely equalled bond will let delaunay disfunction!
            p2d = pa.static_points_analysis_2d(pointsb, hide_figure=False)
            p2d.get_first_minima_bond_length_distribution(
                lattice_constant=1)  # png_filename='bond_hist.png'
            bpm = pa.bond_plot_module(fig, ax)
            bpm.restrict_axis_property_relative(hide_axis=True)
            list_bond_index = bpm.get_bonds_with_conditional_bond_length(
                p2d.bond_length, [0.8, p2d.bond_first_minima_left])
            bpm.draw_points_with_given_bonds(
                pointsb, list_bond_index, bond_color='silver', particle_size=1)
            del atb
        ax.set_aspect('equal', 'box')
        ax.set_xlim([-xylim, xylim])
        ax.set_ylim([-xylim, xylim])
        fig.savefig(png_filename, bbox_inches='tight')
        plt.close('all')
        del at

    def show_dual_type_n_part_special(self, type_n, xylim=5, bond_on=True, n_plus=2):
        R"""
        type_n: if 3, means the polygon is of type_3.(see archimedean_tilings)
        xylim: if 5, means the plot would be 10*10.

        examples:
            import workflow_analysis as wa
            sdl = wa.show_dual_lattice()
            for i in [3,6,7,8,9,10,11]:
                sdl.show_dual_type_n_part(type_n=i,xylim=3,n_plus=2)
        """
        at = archimedean_tilings()
        at.generate_type_n_part(type_n)  # <delta>
        png_filename = 'dual_type'+str(type_n)+'_part_special_bond.png'  # <delta>
        vec = at.a1+at.a2
        n1 = int(np.around(2*xylim/vec[0], 0)+n_plus)
        n2 = int(np.around(2*xylim/vec[1], 0)+n_plus)
        points = at.generate_lattices([n1, n2])  # 1.73:2
        dula = at.get_dual_lattice(points)
        fig, ax = plt.subplots()
        ax.scatter(points[:, 0], points[:, 1], color='k', zorder=3)
        ax.scatter(dula[:, 0], dula[:, 1], facecolors='white', edgecolors='k', zorder=3)
        # draw bonds selected
        if bond_on:
            atb = archimedean_tilings()
            atb.generate_type_n(type_n)  # <delta>
            pointsb = atb.generate_lattices([2*n1, 2*n2])
            perturbation = np.random.random(pointsb.shape)*0.01
            pointsb = pointsb + perturbation  # precisely equalled bond will let delaunay disfunction!
            p2d = pa.static_points_analysis_2d(pointsb, hide_figure=False)
            p2d.get_first_minima_bond_length_distribution(
                lattice_constant=1)  # png_filename='bond_hist.png'
            bpm = pa.bond_plot_module(fig, ax)
            bpm.restrict_axis_property_relative(hide_axis=True)
            list_bond_index = bpm.get_bonds_with_conditional_bond_length(
                p2d.bond_length, [0.8, p2d.bond_first_minima_left])
            bpm.draw_points_with_given_bonds(
                pointsb, list_bond_index, bond_color='silver', particle_size=1)
            del atb
        ax.set_aspect('equal', 'box')
        ax.set_xlim([-xylim, xylim])
        ax.set_ylim([-xylim, xylim])
        fig.savefig(png_filename, bbox_inches='tight')
        plt.close('all')
        del at
