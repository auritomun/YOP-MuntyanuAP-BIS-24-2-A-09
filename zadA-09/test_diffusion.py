import unittest
import numpy as np



def calculate_rms(x, y):
    msd = np.mean(x ** 2 + y ** 2)
    return np.sqrt(msd)


class TestDiffusionSimulator(unittest.TestCase):

    def test_rms_calculation(self):

        x = np.array([10, -10, 10, -10])
        y = np.array([10, 10, -10, -10])

        expected_rms = np.sqrt(200)


        actual_rms = calculate_rms(x, y)


        self.assertAlmostEqual(actual_rms, expected_rms, places=5,
                               msg="Расчет RMS выполнен неверно!")

    def test_edge_case_single_particle(self):

        x = np.array([0])
        y = np.array([0])
        expected_rms = 0.0


        actual_rms = calculate_rms(x, y)


        self.assertEqual(actual_rms, expected_rms,
                         msg="Для одной частицы RMS должен быть равен 0!")


if __name__ == '__main__':
    unittest.main()