from typing import List, Tuple

from warp.fem.polynomial import Polynomial, quadrature_1d
from warp.fem.types import Coords


class Element:
    dimension = 0
    """Intrinsic dimension of the element"""

    def measure() -> float:
        """Measure (area, volume, ...) of the reference element"""
        raise NotImplementedError

    @staticmethod
    def instantiate_quadrature(order: int, family: Polynomial) -> Tuple[List[Coords], List[float]]:
        """Returns a quadrature of a given order for a prototypical element"""
        raise NotImplementedError

    def center(self) -> Tuple[float]:
        coords, _ = self.instantiate_quadrature(order=0, family=None)
        return coords[0]


def _point_count_from_order(order: int, family: Polynomial):
    if family == Polynomial.GAUSS_LEGENDRE:
        point_count = max(1, order // 2 + 1)
    elif family == Polynomial.LOBATTO_GAUSS_LEGENDRE:
        point_count = max(2, order // 2 + 2)
    elif family == Polynomial.EQUISPACED_CLOSED:
        point_count = max(2, 2 * (order // 2) + 1)
    elif family == Polynomial.EQUISPACED_OPEN:
        point_count = max(1, 2 * (order // 2) + 1)

    return point_count


class Cube(Element):
    dimension = 3

    @staticmethod
    def measure() -> float:
        return 1.0

    @staticmethod
    def instantiate_quadrature(order: int, family: Polynomial):
        if family is None:
            family = Polynomial.GAUSS_LEGENDRE

        point_count = _point_count_from_order(order=order, family=family)
        gauss_1d, weights_1d = quadrature_1d(point_count=point_count, family=family)

        coords = [Coords(x, y, z) for x in gauss_1d for y in gauss_1d for z in gauss_1d]
        weights = [wx * wy * wz for wx in weights_1d for wy in weights_1d for wz in weights_1d]

        return coords, weights


class Square(Element):
    dimension = 2

    @staticmethod
    def measure() -> float:
        return 1.0

    @staticmethod
    def instantiate_quadrature(order: int, family: Polynomial):
        if family is None:
            family = Polynomial.GAUSS_LEGENDRE

        point_count = _point_count_from_order(order=order, family=family)
        gauss_1d, weights_1d = quadrature_1d(point_count=point_count, family=family)

        coords = [Coords(x, y, 0.0) for x in gauss_1d for y in gauss_1d]
        weights = [wx * wy for wx in weights_1d for wy in weights_1d]

        return coords, weights


class LinearEdge(Element):
    dimension = 1

    @staticmethod
    def measure() -> float:
        return 1.0

    @staticmethod
    def instantiate_quadrature(order: int, family: Polynomial):
        if family is None:
            family = Polynomial.GAUSS_LEGENDRE

        point_count = _point_count_from_order(order=order, family=family)
        gauss_1d, weights_1d = quadrature_1d(point_count=point_count, family=family)

        coords = [Coords(x, 0.0, 0.0) for x in gauss_1d]
        return coords, weights_1d


class Triangle(Element):
    dimension = 2

    @staticmethod
    def measure() -> float:
        return 0.5

    @staticmethod
    def instantiate_quadrature(order: int, family: Polynomial):
        if family is not None:
            # Duffy transformation from square to triangle
            point_count = _point_count_from_order(order=order + 1, family=family)
            gauss_1d, weights_1d = quadrature_1d(point_count=point_count, family=family)

            coords = [Coords(1.0 - x - y + x * y, x, y * (1.0 - x)) for x in gauss_1d for y in gauss_1d]

            # Scale weight by 2.0 so that they sum up to 1
            weights = [2.0 * wx * (1.0 - x) * wy for x, wx in zip(gauss_1d, weights_1d) for wy in weights_1d]

            return coords, weights

        if order <= 1:
            weights = [1.0]
            coords = [Coords(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)]
        elif order <= 2:
            weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
            coords = [
                Coords(2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
                Coords(1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
                Coords(1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
            ]
        elif order <= 3:
            # Hillion 1977,
            # "Numerical Integration on a Triangle"
            weights = [
                3.18041381743977225049491153185954e-01,
                3.18041381743977225049491153185954e-01,
                1.81958618256022719439357615556219e-01,
                1.81958618256022719439357615556219e-01,
            ]

            coords = [
                Coords(
                    6.66390246014701426169324349757517e-01,
                    1.78558728263616461884311092944699e-01,
                    1.55051025721682111946364557297784e-01,
                ),
                Coords(
                    1.78558728263616461884311092944699e-01,
                    6.66390246014701426169324349757517e-01,
                    1.55051025721682056435213326039957e-01,
                ),
                Coords(
                    2.80019915499074012465996474929852e-01,
                    7.50311102226081383381739442484104e-02,
                    6.44948974278317876951405196450651e-01,
                ),
                Coords(
                    7.50311102226081383381739442484104e-02,
                    2.80019915499074012465996474929852e-01,
                    6.44948974278317876951405196450651e-01,
                ),
            ]
        elif order <= 4:
            # Witherden and Vincent 2015,
            # "On the identification of symmetric quadrature rules for finite element methods"
            # https://doi.org/10.1016/j.camwa.2015.03.017

            weights = [
                2.23381589678011471811203136894619e-01,
                2.23381589678011471811203136894619e-01,
                2.23381589678011471811203136894619e-01,
                1.09951743655321870773988734981685e-01,
                1.09951743655321870773988734981685e-01,
                1.09951743655321870773988734981685e-01,
            ]

            coords = [
                Coords(
                    4.45948490915964890213274429697776e-01,
                    4.45948490915964890213274429697776e-01,
                    1.08103018168070219573451140604448e-01,
                ),
                Coords(
                    4.45948490915964890213274429697776e-01,
                    1.08103018168070219573451140604448e-01,
                    4.45948490915964890213274429697776e-01,
                ),
                Coords(
                    1.08103018168070219573451140604448e-01,
                    4.45948490915964890213274429697776e-01,
                    4.45948490915964890213274429697776e-01,
                ),
                Coords(
                    9.15762135097707430375635340169538e-02,
                    9.15762135097707430375635340169538e-02,
                    8.16847572980458513924872931966092e-01,
                ),
                Coords(
                    9.15762135097707430375635340169538e-02,
                    8.16847572980458513924872931966092e-01,
                    9.15762135097707430375635340169538e-02,
                ),
                Coords(
                    8.16847572980458513924872931966092e-01,
                    9.15762135097707430375635340169538e-02,
                    9.15762135097707430375635340169538e-02,
                ),
            ]

        elif order <= 5:
            weights = [
                2.25000000000000005551115123125783e-01,
                1.25939180544827139529573400977824e-01,
                1.25939180544827139529573400977824e-01,
                1.25939180544827139529573400977824e-01,
                1.32394152788506191953388224646915e-01,
                1.32394152788506191953388224646915e-01,
                1.32394152788506191953388224646915e-01,
            ]

            coords = [
                Coords(
                    3.33333333333333314829616256247391e-01,
                    3.33333333333333314829616256247391e-01,
                    3.33333333333333314829616256247391e-01,
                ),
                Coords(
                    1.01286507323456342888334802410100e-01,
                    1.01286507323456342888334802410100e-01,
                    7.97426985353087314223330395179801e-01,
                ),
                Coords(
                    1.01286507323456342888334802410100e-01,
                    7.97426985353087314223330395179801e-01,
                    1.01286507323456342888334802410100e-01,
                ),
                Coords(
                    7.97426985353087314223330395179801e-01,
                    1.01286507323456342888334802410100e-01,
                    1.01286507323456342888334802410100e-01,
                ),
                Coords(
                    4.70142064105115109473587153843255e-01,
                    4.70142064105115109473587153843255e-01,
                    5.97158717897697810528256923134904e-02,
                ),
                Coords(
                    4.70142064105115109473587153843255e-01,
                    5.97158717897697810528256923134904e-02,
                    4.70142064105115109473587153843255e-01,
                ),
                Coords(
                    5.97158717897697810528256923134904e-02,
                    4.70142064105115109473587153843255e-01,
                    4.70142064105115109473587153843255e-01,
                ),
            ]
        elif order <= 6:
            weights = [
                5.08449063702068326797700592578622e-02,
                5.08449063702068326797700592578622e-02,
                5.08449063702068326797700592578622e-02,
                1.16786275726379396022736045779311e-01,
                1.16786275726379396022736045779311e-01,
                1.16786275726379396022736045779311e-01,
                8.28510756183735846969184990484791e-02,
                8.28510756183735846969184990484791e-02,
                8.28510756183735846969184990484791e-02,
                8.28510756183735846969184990484791e-02,
                8.28510756183735846969184990484791e-02,
                8.28510756183735846969184990484791e-02,
            ]

            coords = [
                Coords(
                    6.30890144915022266225435032538371e-02,
                    6.30890144915022266225435032538371e-02,
                    8.73821971016995546754912993492326e-01,
                ),
                Coords(
                    6.30890144915022266225435032538371e-02,
                    8.73821971016995546754912993492326e-01,
                    6.30890144915022266225435032538371e-02,
                ),
                Coords(
                    8.73821971016995546754912993492326e-01,
                    6.30890144915022266225435032538371e-02,
                    6.30890144915022266225435032538371e-02,
                ),
                Coords(
                    2.49286745170910428726074314909056e-01,
                    2.49286745170910428726074314909056e-01,
                    5.01426509658179142547851370181888e-01,
                ),
                Coords(
                    2.49286745170910428726074314909056e-01,
                    5.01426509658179142547851370181888e-01,
                    2.49286745170910428726074314909056e-01,
                ),
                Coords(
                    5.01426509658179142547851370181888e-01,
                    2.49286745170910428726074314909056e-01,
                    2.49286745170910428726074314909056e-01,
                ),
                Coords(
                    5.31450498448169383891581674106419e-02,
                    3.10352451033784393352732422499685e-01,
                    6.36502499121398668258109410089673e-01,
                ),
                Coords(
                    5.31450498448169383891581674106419e-02,
                    6.36502499121398668258109410089673e-01,
                    3.10352451033784393352732422499685e-01,
                ),
                Coords(
                    3.10352451033784393352732422499685e-01,
                    5.31450498448169383891581674106419e-02,
                    6.36502499121398668258109410089673e-01,
                ),
                Coords(
                    3.10352451033784393352732422499685e-01,
                    6.36502499121398668258109410089673e-01,
                    5.31450498448169383891581674106419e-02,
                ),
                Coords(
                    6.36502499121398668258109410089673e-01,
                    5.31450498448169383891581674106419e-02,
                    3.10352451033784393352732422499685e-01,
                ),
                Coords(
                    6.36502499121398668258109410089673e-01,
                    3.10352451033784393352732422499685e-01,
                    5.31450498448169383891581674106419e-02,
                ),
            ]
        else:
            # Order 8

            weights = [
                1.44315607677787172136163462710101e-01,
                9.50916342672846193195823616406415e-02,
                9.50916342672846193195823616406415e-02,
                9.50916342672846193195823616406415e-02,
                1.03217370534718244634575512463925e-01,
                1.03217370534718244634575512463925e-01,
                1.03217370534718244634575512463925e-01,
                3.24584976231980792960030157701112e-02,
                3.24584976231980792960030157701112e-02,
                3.24584976231980792960030157701112e-02,
                2.72303141744349927466650740370824e-02,
                2.72303141744349927466650740370824e-02,
                2.72303141744349927466650740370824e-02,
                2.72303141744349927466650740370824e-02,
                2.72303141744349927466650740370824e-02,
                2.72303141744349927466650740370824e-02,
            ]

            coords = [
                Coords(
                    3.33333333333333314829616256247391e-01,
                    3.33333333333333314829616256247391e-01,
                    3.33333333333333314829616256247391e-01,
                ),
                Coords(
                    4.59292588292723125142913431773195e-01,
                    4.59292588292723125142913431773195e-01,
                    8.14148234145537497141731364536099e-02,
                ),
                Coords(
                    4.59292588292723125142913431773195e-01,
                    8.14148234145537497141731364536099e-02,
                    4.59292588292723125142913431773195e-01,
                ),
                Coords(
                    8.14148234145537497141731364536099e-02,
                    4.59292588292723125142913431773195e-01,
                    4.59292588292723125142913431773195e-01,
                ),
                Coords(
                    1.70569307751760212976677166807349e-01,
                    1.70569307751760212976677166807349e-01,
                    6.58861384496479574046645666385302e-01,
                ),
                Coords(
                    1.70569307751760212976677166807349e-01,
                    6.58861384496479574046645666385302e-01,
                    1.70569307751760212976677166807349e-01,
                ),
                Coords(
                    6.58861384496479574046645666385302e-01,
                    1.70569307751760212976677166807349e-01,
                    1.70569307751760212976677166807349e-01,
                ),
                Coords(
                    5.05472283170309566457945038564503e-02,
                    5.05472283170309566457945038564503e-02,
                    8.98905543365938086708410992287099e-01,
                ),
                Coords(
                    5.05472283170309566457945038564503e-02,
                    8.98905543365938086708410992287099e-01,
                    5.05472283170309566457945038564503e-02,
                ),
                Coords(
                    8.98905543365938086708410992287099e-01,
                    5.05472283170309566457945038564503e-02,
                    5.05472283170309566457945038564503e-02,
                ),
                Coords(
                    8.39477740995758781039626228448469e-03,
                    2.63112829634638112352718053443823e-01,
                    7.28492392955404355348036915529519e-01,
                ),
                Coords(
                    8.39477740995758781039626228448469e-03,
                    7.28492392955404355348036915529519e-01,
                    2.63112829634638112352718053443823e-01,
                ),
                Coords(
                    2.63112829634638112352718053443823e-01,
                    8.39477740995758781039626228448469e-03,
                    7.28492392955404355348036915529519e-01,
                ),
                Coords(
                    2.63112829634638112352718053443823e-01,
                    7.28492392955404355348036915529519e-01,
                    8.39477740995758781039626228448469e-03,
                ),
                Coords(
                    7.28492392955404355348036915529519e-01,
                    8.39477740995758781039626228448469e-03,
                    2.63112829634638112352718053443823e-01,
                ),
                Coords(
                    7.28492392955404355348036915529519e-01,
                    2.63112829634638112352718053443823e-01,
                    8.39477740995758781039626228448469e-03,
                ),
            ]

        return coords, weights


class Tetrahedron(Element):
    dimension = 3

    @staticmethod
    def measure() -> float:
        return 1.0 / 6.0

    @staticmethod
    def instantiate_quadrature(order: int, family: Polynomial):
        if family is not None:
            # Duffy transformation from square to triangle
            point_count = _point_count_from_order(order=order + 1, family=family)
            gauss_1d, weights_1d = quadrature_1d(point_count=point_count, family=family)

            coords = [
                Coords(x, y * (1.0 - x), z * (1.0 - x) * (1.0 - y))
                for x in gauss_1d
                for y in gauss_1d
                for z in gauss_1d
            ]

            # Scale weight by 6.0 so that they sum up to 1
            weights = [
                6.0 * wx * wy * wz * (1.0 - x) * (1.0 - x) * (1.0 - y)
                for x, wx in zip(gauss_1d, weights_1d)
                for y, wy in zip(gauss_1d, weights_1d)
                for wz in weights_1d
            ]

            return coords, weights

        # Shunn and Ham 2012
        # "Symmetric quadrature rules for tetrahedra based on a cubic close-packed lattice arrangement"
        # https://doi.org/10.1016/j.cam.2012.03.032

        # TODO: add Witherden and Vincent 2015,

        if order <= 1:
            weights = [1.0]
            coords = [Coords(1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0)]
        elif order <= 2:
            weights = [1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0]
            coords = [
                Coords(0.1381966011250110, 0.1381966011250110, 0.1381966011250110),
                Coords(0.5854101966249680, 0.1381966011250110, 0.1381966011250110),
                Coords(0.1381966011250110, 0.5854101966249680, 0.1381966011250110),
                Coords(0.1381966011250110, 0.1381966011250110, 0.5854101966249680),
            ]
        elif order <= 3:
            weights = [
                0.0476331348432089,
                0.0476331348432089,
                0.0476331348432089,
                0.0476331348432089,
                0.1349112434378610,
                0.1349112434378610,
                0.1349112434378610,
                0.1349112434378610,
                0.1349112434378610,
                0.1349112434378610,
            ]

            coords = [
                Coords(0.0738349017262234, 0.0738349017262234, 0.0738349017262234),
                Coords(0.7784952948213300, 0.0738349017262234, 0.0738349017262234),
                Coords(0.0738349017262234, 0.7784952948213300, 0.0738349017262234),
                Coords(0.0738349017262234, 0.0738349017262234, 0.7784952948213300),
                Coords(0.4062443438840510, 0.0937556561159491, 0.0937556561159491),
                Coords(0.0937556561159491, 0.4062443438840510, 0.0937556561159491),
                Coords(0.0937556561159491, 0.0937556561159491, 0.4062443438840510),
                Coords(0.4062443438840510, 0.4062443438840510, 0.0937556561159491),
                Coords(0.4062443438840510, 0.0937556561159491, 0.4062443438840510),
                Coords(0.0937556561159491, 0.4062443438840510, 0.4062443438840510),
            ]
        elif order <= 4:
            weights = [
                0.0070670747944695,
                0.0070670747944695,
                0.0070670747944695,
                0.0070670747944695,
                0.0469986689718877,
                0.0469986689718877,
                0.0469986689718877,
                0.0469986689718877,
                0.0469986689718877,
                0.0469986689718877,
                0.0469986689718877,
                0.0469986689718877,
                0.0469986689718877,
                0.0469986689718877,
                0.0469986689718877,
                0.0469986689718877,
                0.1019369182898680,
                0.1019369182898680,
                0.1019369182898680,
                0.1019369182898680,
            ]

            coords = [
                Coords(0.0323525947272439, 0.0323525947272439, 0.0323525947272439),
                Coords(0.9029422158182680, 0.0323525947272439, 0.0323525947272439),
                Coords(0.0323525947272439, 0.9029422158182680, 0.0323525947272439),
                Coords(0.0323525947272439, 0.0323525947272439, 0.9029422158182680),
                Coords(0.6165965330619370, 0.0603604415251421, 0.0603604415251421),
                Coords(0.2626825838877790, 0.0603604415251421, 0.0603604415251421),
                Coords(0.0603604415251421, 0.6165965330619370, 0.0603604415251421),
                Coords(0.0603604415251421, 0.2626825838877790, 0.0603604415251421),
                Coords(0.0603604415251421, 0.0603604415251421, 0.6165965330619370),
                Coords(0.0603604415251421, 0.0603604415251421, 0.2626825838877790),
                Coords(0.2626825838877790, 0.6165965330619370, 0.0603604415251421),
                Coords(0.6165965330619370, 0.2626825838877790, 0.0603604415251421),
                Coords(0.2626825838877790, 0.0603604415251421, 0.6165965330619370),
                Coords(0.6165965330619370, 0.0603604415251421, 0.2626825838877790),
                Coords(0.0603604415251421, 0.2626825838877790, 0.6165965330619370),
                Coords(0.0603604415251421, 0.6165965330619370, 0.2626825838877790),
                Coords(0.3097693042728620, 0.3097693042728620, 0.0706920871814129),
                Coords(0.3097693042728620, 0.0706920871814129, 0.3097693042728620),
                Coords(0.0706920871814129, 0.3097693042728620, 0.3097693042728620),
                Coords(0.3097693042728620, 0.3097693042728620, 0.3097693042728620),
            ]

        elif order <= 5:
            weights = [
                0.0021900463965388,
                0.0021900463965388,
                0.0021900463965388,
                0.0021900463965388,
                0.0143395670177665,
                0.0143395670177665,
                0.0143395670177665,
                0.0143395670177665,
                0.0143395670177665,
                0.0143395670177665,
                0.0143395670177665,
                0.0143395670177665,
                0.0143395670177665,
                0.0143395670177665,
                0.0143395670177665,
                0.0143395670177665,
                0.0250305395686746,
                0.0250305395686746,
                0.0250305395686746,
                0.0250305395686746,
                0.0250305395686746,
                0.0250305395686746,
                0.0479839333057554,
                0.0479839333057554,
                0.0479839333057554,
                0.0479839333057554,
                0.0479839333057554,
                0.0479839333057554,
                0.0479839333057554,
                0.0479839333057554,
                0.0479839333057554,
                0.0479839333057554,
                0.0479839333057554,
                0.0479839333057554,
                0.0931745731195340,
            ]

            coords = [
                Coords(0.0267367755543735, 0.0267367755543735, 0.0267367755543735),
                Coords(0.9197896733368800, 0.0267367755543735, 0.0267367755543735),
                Coords(0.0267367755543735, 0.9197896733368800, 0.0267367755543735),
                Coords(0.0267367755543735, 0.0267367755543735, 0.9197896733368800),
                Coords(0.7477598884818090, 0.0391022406356488, 0.0391022406356488),
                Coords(0.1740356302468940, 0.0391022406356488, 0.0391022406356488),
                Coords(0.0391022406356488, 0.7477598884818090, 0.0391022406356488),
                Coords(0.0391022406356488, 0.1740356302468940, 0.0391022406356488),
                Coords(0.0391022406356488, 0.0391022406356488, 0.7477598884818090),
                Coords(0.0391022406356488, 0.0391022406356488, 0.1740356302468940),
                Coords(0.1740356302468940, 0.7477598884818090, 0.0391022406356488),
                Coords(0.7477598884818090, 0.1740356302468940, 0.0391022406356488),
                Coords(0.1740356302468940, 0.0391022406356488, 0.7477598884818090),
                Coords(0.7477598884818090, 0.0391022406356488, 0.1740356302468940),
                Coords(0.0391022406356488, 0.1740356302468940, 0.7477598884818090),
                Coords(0.0391022406356488, 0.7477598884818090, 0.1740356302468940),
                Coords(0.4547545999844830, 0.0452454000155172, 0.0452454000155172),
                Coords(0.0452454000155172, 0.4547545999844830, 0.0452454000155172),
                Coords(0.0452454000155172, 0.0452454000155172, 0.4547545999844830),
                Coords(0.4547545999844830, 0.4547545999844830, 0.0452454000155172),
                Coords(0.4547545999844830, 0.0452454000155172, 0.4547545999844830),
                Coords(0.0452454000155172, 0.4547545999844830, 0.4547545999844830),
                Coords(0.2232010379623150, 0.2232010379623150, 0.0504792790607720),
                Coords(0.5031186450145980, 0.2232010379623150, 0.0504792790607720),
                Coords(0.2232010379623150, 0.5031186450145980, 0.0504792790607720),
                Coords(0.2232010379623150, 0.0504792790607720, 0.2232010379623150),
                Coords(0.5031186450145980, 0.0504792790607720, 0.2232010379623150),
                Coords(0.2232010379623150, 0.0504792790607720, 0.5031186450145980),
                Coords(0.0504792790607720, 0.2232010379623150, 0.2232010379623150),
                Coords(0.0504792790607720, 0.5031186450145980, 0.2232010379623150),
                Coords(0.0504792790607720, 0.2232010379623150, 0.5031186450145980),
                Coords(0.5031186450145980, 0.2232010379623150, 0.2232010379623150),
                Coords(0.2232010379623150, 0.5031186450145980, 0.2232010379623150),
                Coords(0.2232010379623150, 0.2232010379623150, 0.5031186450145980),
                Coords(0.2500000000000000, 0.2500000000000000, 0.2500000000000000),
            ]
        elif order <= 6:
            weights = [
                0.0010373112336140,
                0.0010373112336140,
                0.0010373112336140,
                0.0010373112336140,
                0.0096016645399480,
                0.0096016645399480,
                0.0096016645399480,
                0.0096016645399480,
                0.0096016645399480,
                0.0096016645399480,
                0.0096016645399480,
                0.0096016645399480,
                0.0096016645399480,
                0.0096016645399480,
                0.0096016645399480,
                0.0096016645399480,
                0.0164493976798232,
                0.0164493976798232,
                0.0164493976798232,
                0.0164493976798232,
                0.0164493976798232,
                0.0164493976798232,
                0.0164493976798232,
                0.0164493976798232,
                0.0164493976798232,
                0.0164493976798232,
                0.0164493976798232,
                0.0164493976798232,
                0.0153747766513310,
                0.0153747766513310,
                0.0153747766513310,
                0.0153747766513310,
                0.0153747766513310,
                0.0153747766513310,
                0.0153747766513310,
                0.0153747766513310,
                0.0153747766513310,
                0.0153747766513310,
                0.0153747766513310,
                0.0153747766513310,
                0.0293520118375230,
                0.0293520118375230,
                0.0293520118375230,
                0.0293520118375230,
                0.0293520118375230,
                0.0293520118375230,
                0.0293520118375230,
                0.0293520118375230,
                0.0293520118375230,
                0.0293520118375230,
                0.0293520118375230,
                0.0293520118375230,
                0.0366291366405108,
                0.0366291366405108,
                0.0366291366405108,
                0.0366291366405108,
            ]

            coords = [
                Coords(0.0149520651530592, 0.0149520651530592, 0.0149520651530592),
                Coords(0.9551438045408220, 0.0149520651530592, 0.0149520651530592),
                Coords(0.0149520651530592, 0.9551438045408220, 0.0149520651530592),
                Coords(0.0149520651530592, 0.0149520651530592, 0.9551438045408220),
                Coords(0.1518319491659370, 0.0340960211962615, 0.0340960211962615),
                Coords(0.7799760084415400, 0.0340960211962615, 0.0340960211962615),
                Coords(0.0340960211962615, 0.1518319491659370, 0.0340960211962615),
                Coords(0.0340960211962615, 0.7799760084415400, 0.0340960211962615),
                Coords(0.0340960211962615, 0.0340960211962615, 0.1518319491659370),
                Coords(0.0340960211962615, 0.0340960211962615, 0.7799760084415400),
                Coords(0.7799760084415400, 0.1518319491659370, 0.0340960211962615),
                Coords(0.1518319491659370, 0.7799760084415400, 0.0340960211962615),
                Coords(0.7799760084415400, 0.0340960211962615, 0.1518319491659370),
                Coords(0.1518319491659370, 0.0340960211962615, 0.7799760084415400),
                Coords(0.0340960211962615, 0.7799760084415400, 0.1518319491659370),
                Coords(0.0340960211962615, 0.1518319491659370, 0.7799760084415400),
                Coords(0.5526556431060170, 0.0462051504150017, 0.0462051504150017),
                Coords(0.3549340560639790, 0.0462051504150017, 0.0462051504150017),
                Coords(0.0462051504150017, 0.5526556431060170, 0.0462051504150017),
                Coords(0.0462051504150017, 0.3549340560639790, 0.0462051504150017),
                Coords(0.0462051504150017, 0.0462051504150017, 0.5526556431060170),
                Coords(0.0462051504150017, 0.0462051504150017, 0.3549340560639790),
                Coords(0.3549340560639790, 0.5526556431060170, 0.0462051504150017),
                Coords(0.5526556431060170, 0.3549340560639790, 0.0462051504150017),
                Coords(0.3549340560639790, 0.0462051504150017, 0.5526556431060170),
                Coords(0.5526556431060170, 0.0462051504150017, 0.3549340560639790),
                Coords(0.0462051504150017, 0.3549340560639790, 0.5526556431060170),
                Coords(0.0462051504150017, 0.5526556431060170, 0.3549340560639790),
                Coords(0.2281904610687610, 0.2281904610687610, 0.0055147549744775),
                Coords(0.5381043228880020, 0.2281904610687610, 0.0055147549744775),
                Coords(0.2281904610687610, 0.5381043228880020, 0.0055147549744775),
                Coords(0.2281904610687610, 0.0055147549744775, 0.2281904610687610),
                Coords(0.5381043228880020, 0.0055147549744775, 0.2281904610687610),
                Coords(0.2281904610687610, 0.0055147549744775, 0.5381043228880020),
                Coords(0.0055147549744775, 0.2281904610687610, 0.2281904610687610),
                Coords(0.0055147549744775, 0.5381043228880020, 0.2281904610687610),
                Coords(0.0055147549744775, 0.2281904610687610, 0.5381043228880020),
                Coords(0.5381043228880020, 0.2281904610687610, 0.2281904610687610),
                Coords(0.2281904610687610, 0.5381043228880020, 0.2281904610687610),
                Coords(0.2281904610687610, 0.2281904610687610, 0.5381043228880020),
                Coords(0.3523052600879940, 0.3523052600879940, 0.0992057202494530),
                Coords(0.1961837595745600, 0.3523052600879940, 0.0992057202494530),
                Coords(0.3523052600879940, 0.1961837595745600, 0.0992057202494530),
                Coords(0.3523052600879940, 0.0992057202494530, 0.3523052600879940),
                Coords(0.1961837595745600, 0.0992057202494530, 0.3523052600879940),
                Coords(0.3523052600879940, 0.0992057202494530, 0.1961837595745600),
                Coords(0.0992057202494530, 0.3523052600879940, 0.3523052600879940),
                Coords(0.0992057202494530, 0.1961837595745600, 0.3523052600879940),
                Coords(0.0992057202494530, 0.3523052600879940, 0.1961837595745600),
                Coords(0.1961837595745600, 0.3523052600879940, 0.3523052600879940),
                Coords(0.3523052600879940, 0.1961837595745600, 0.3523052600879940),
                Coords(0.3523052600879940, 0.3523052600879940, 0.1961837595745600),
                Coords(0.1344783347929940, 0.1344783347929940, 0.1344783347929940),
                Coords(0.5965649956210170, 0.1344783347929940, 0.1344783347929940),
                Coords(0.1344783347929940, 0.5965649956210170, 0.1344783347929940),
                Coords(0.1344783347929940, 0.1344783347929940, 0.5965649956210170),
            ]
        else:
            raise NotImplementedError

        return coords, weights
