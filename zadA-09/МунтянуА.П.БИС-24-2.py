import numpy as np
import matplotlib.pyplot as plt
import math


N = 1000
steps = 100
step_size = 1.0

angles = np.random.uniform(0, 2 * np.pi, (steps, N))

dx = step_size * np.cos(angles)
dy = step_size * np.sin(angles)

x = np.zeros((steps + 1, N))
y = np.zeros((steps + 1, N))
x[1:, :] = np.cumsum(dx, axis=0)
y[1:, :] = np.cumsum(dy, axis=0)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for i in range(50):  # Рисуем первые 50 частиц, чтобы не перегружать график
    plt.plot(x[:, i], y[:, i], alpha=0.6, linewidth=0.8)
plt.title("Траектории броуновского движения (первые 50 частиц)")
plt.xlabel("Координата X")
plt.ylabel("Координата Y")
plt.axis('equal')
plt.grid(True, alpha=0.3)

final_r = np.sqrt(x[-1, :]**2 + y[-1, :]**2)

plt.subplot(1, 2, 2)
plt.hist(final_r, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.title("Гистограмма финального распределения частиц (шаг {})".format(steps))
plt.xlabel("Расстояние от центра (R)")
plt.ylabel("Плотность вероятности")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

msd = np.mean(x**2 + y**2, axis=1)

rms_deviation = np.sqrt(msd[-1])

theoretical_msd = (step_size**2) * steps
theoretical_rms = np.sqrt(theoretical_msd)

data_to_export = np.column_stack((x[-1, :], y[-1, :], final_r))
np.savetxt("diffusion_simulation_results.csv", data_to_export,
           delimiter=",", header="X_final, Y_final, R_final", comments="")

print("=== РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ ===")
print(f"Количество частиц (N): {N}")
print(f"Количество шагов (t): {steps}")
print(f"Среднеквадратичное отклонение (RMS - Модель): {rms_deviation:.4f}")
print(f"Среднеквадратичное отклонение (RMS - Теория): {theoretical_rms:.4f}")
print("Данные экспортированы в файл: diffusion_simulation_results.csv")