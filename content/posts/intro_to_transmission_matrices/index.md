---
title: 'Introduction to Transmission Matrices'
date: 2024-04-01T01:06:37-03:00
---

# Introducción a matrices de transmisión

## Cuadripolos
Las matrices de transmisión surgen a partir de las ecuaciones que caracterízan a los cuadripolos. De forma general, la matriz que caracteriza la transmisión de un cuadripolo se puede definir como:

<div>
$$
    T =\begin{bmatrix}
    t_{11} & t_{12} \\
    t_{21} & t_{22}
    \end{bmatrix}
$$
</div>

donde cada uno de los valores de la matriz $T$ representa una relación especifica entre la entrada y la salida del cuadripolo.

<div>
$$
    \begin{align*}
        t_{11} &= \frac{v_{in}}{v_{out}} \bigg|_{i_{out} = 0} \hspace{8mm} \rightarrow \hspace{8mm} (1 / G_v)\\
        t_{12} &= \frac{v_{in}}{i_{out}} \bigg|_{i_{out} = 0} \hspace{8mm} \rightarrow \hspace{8mm} (1 / G_Y)\\
        t_{21} &= \frac{i_{in}}{v_{out}} \bigg|_{v_{out} = 0} \hspace{8mm} \rightarrow \hspace{8mm} (1 / G_Z)\\
        t_{22} &= \frac{i_{in}}{i_{out}} \bigg|_{v_{out} = 0} \hspace{8mm} \rightarrow \hspace{8mm} (1 / G_i)\\
    \end{align*}
$$
</div>


Como se puede ver, hay dos condiciones sobre las cuales se especifican los elementos.
- $v_{out} = 0$: Significa que la tensión de salida es nula, por lo que los conectores de salida estan cortocircuitados.
- $i_{out} = 0$: Significa que la corriente de salida es nula, por lo que los conectores de salida estan abiertos.

Dicha matriz nos dice como obtener las variables de entrada en función de las variables de salida:

<div>
$$
    \begin{equation*}
        \begin{bmatrix}
            v_{in} \\
            i_{in}
        \end{bmatrix}
        =
        T \cdot
        \begin{bmatrix}
            v_{out} \\
            i_{out}
        \end{bmatrix}
        =
        \begin{bmatrix}
            t_{11} & t_{12} \\
            t_{21} & t_{22}
        \end{bmatrix}
        \cdot
        \begin{bmatrix}
            v_{out} \\
            i_{out}
        \end{bmatrix}
        =
        \begin{bmatrix}
            t_{11} \cdot v_{out} + t_{12} \cdot i_{out}\\
            t_{21} \cdot v_{out} + t_{22} \cdot i_{out}
        \end{bmatrix}
    \end{equation*}
$$
</div>

Por lo que para obtener los variables de salida en función de las variables de entrada, debemos invertir la matriz de transmisión y aplicarla a las variables de entrada.

<div>
$$
    T^{-1} = 
    \frac{1}{\det(T)} \cdot
    \begin{bmatrix}
        t_{22} & -t_{12} \\
        -t_{21} & t_{11}
    \end{bmatrix}
$$
</div>
<div>
$$
    \begin{bmatrix}
        v_{in} \\
        i_{in}
    \end{bmatrix}
    =
    T \cdot
    \begin{bmatrix}
        v_{out} \\
        i_{out}
    \end{bmatrix}
    \hspace{4mm} \rightarrow \hspace{4mm}
    T^{-1} \cdot
    \begin{bmatrix}
        v_{in} \\
        i_{in}
    \end{bmatrix}
    =
    \begin{bmatrix}
        v_{out} \\
        i_{out}
    \end{bmatrix}
    =
    \begin{bmatrix}
        t_{22} \cdot v_{in} - t_{12} \cdot i_{in}\\
        -t_{21} \cdot v_{in} + t_{11} \cdot i_{in}
    \end{bmatrix}
    \cdot \frac{1}{\det(T)}
$$
</div>

## Matrices de transmisión

Cada componente que compone un circuito (ya sea eléctrico, mecánico o acústico) tiene su representación a través de un cuadripolo en forma matricial. De forma general, la representación matricial se diferencia si los componentes son de impedancia, admitancia o transformación.

**Matriz de transmisión de una impedancia ($T_Z$)**
<div>
$$
T_Z =
\begin{bmatrix}
    1 & Z \\
    0 & 1
\end{bmatrix}
$$
</div>

**Matriz de transmisión de una admitancia ($T_Y$)**
<div>
$$
T_Y =
\begin{bmatrix}
    1 & 0 \\
    Y & 1
\end{bmatrix}
$$
</div>

**Matriz de transmisión de un transformador ($T_T$)**
<div>
$$
T_T =
\begin{bmatrix}
    n & 0 \\
    0 & 1/n
\end{bmatrix}
$$
</div>

**Matriz de transmisión de un girador ($T_G$)**
<div>
$$
T_G =
\begin{bmatrix}
    0 & n \\
    1/n & 0
\end{bmatrix}
$$
</div>

## Ejemplo 1: Divisor de tensión

Supongamos que vamos a analizar el siguiente divisor resistivo a través de las matrices de transmisión de cada componente para así obtener la transmisión total del sistema.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/d/db/Resistive_divider.png" width="30%">

</p>

La idea es obtener una ecuación que relacione las magnitudes de salida (tensión y corriente de salida) en función de las magnitudes de entrada (tensión y corriente de entrada).

Primero se deben definir las matrices de ambos elementos. En este caso, $R_1$ está en serie, pero $R_2$ está en paralelo a la salida, por lo que su matriz de transmisión debe ser del tipo admitancia.

<div>
$$
    T_{R_1} =
    \begin{bmatrix}
        1 & R_1 \\
        0 & 1
    \end{bmatrix}
$$
</div>

<div>
$$
    T_{R_2} =
    \begin{bmatrix}
        1 & 0 \\
        \frac{1}{R_2} & 1
    \end{bmatrix}
$$
</div>

Realizando la multiplicación sucesiva de matrices de transmisión, la transferencia total será:

<div>
$$
    T = T_{R_1} \cdot T_{R_2} = 
    \begin{bmatrix}
        1 & R_1 \\
        0 & 1
    \end{bmatrix}
    \cdot
    \begin{bmatrix}
        1 & 0 \\
        \frac{1}{R_2} & 1
    \end{bmatrix}
    =
    \begin{bmatrix}
        1 + \frac{R_1}{R_2} & R_1 \\
        \frac{1}{R_2} & 1
    \end{bmatrix}
$$
</div>

Recordando las ecuaciones descritas anteriormente, la relación entre las magnitudes de entrada y salida queda:

<div>
$$
    \begin{equation*}
        \begin{bmatrix}
            v_{in} \\
            i_{in}
        \end{bmatrix}
        =
        T \cdot
        \begin{bmatrix}
            v_{out} \\
            i_{out}
        \end{bmatrix}
        =
        \begin{bmatrix}
          1 + \frac{R_1}{R_2} & R_1 \\
          \frac{1}{R_2} & 1
        \end{bmatrix}
        \cdot
        \begin{bmatrix}
            v_{out} \\
            i_{out}
        \end{bmatrix}
        =
        \begin{bmatrix}
            (1 + \frac{R_1}{R_2}) \cdot v_{out} + R_{1} \cdot i_{out}\\
            \frac{1}{R_2} \cdot v_{out} + i_{out}
        \end{bmatrix}
    \end{equation*}
$$
</div>

Pero lo que nos importa en la mayoría de los casos, es obtener las magnitudes de salida en función de las de entrada, por lo que:

<div>
$$
    T^{-1} = 
    \begin{bmatrix}
        1 & -R_1 \\
        -\frac{1}{R_2} & 1 + \frac{R_1}{R_2}
    \end{bmatrix}
    \cdot \underbrace{\det(T)}_{=1} =
    \begin{bmatrix}
        1 & -R_1 \\
        -\frac{1}{R_2} & 1 + \frac{R_1}{R_2}
    \end{bmatrix}
$$
</div>

Ahora si podemos obtener la tensión y corriente de salida en función de la tensión y corriente de entrada.

<div>
$$
    T^{-1} \cdot
    \begin{bmatrix}
        v_{in} \\
        i_{in}
    \end{bmatrix}
    =
    \begin{bmatrix}
        1 & -R_1 \\
        -\frac{1}{R_2} & 1 + \frac{R_1}{R_2}
    \end{bmatrix}
    \cdot
    \begin{bmatrix}
        v_{in} \\
        i_{in}
    \end{bmatrix}
    =
    \begin{bmatrix}
        v_{out} \\
        i_{out}
    \end{bmatrix}
    =
    \begin{bmatrix}
        v_{in} - R_{1} \cdot i_{in}\\
        -\frac{1}{R_2} \cdot v_{in} + (1 + \frac{R_1}{R_2}) \cdot i_{in}
    \end{bmatrix}
$$
</div>

<div>
$$
    \begin{align*}
        v_{out} = v_{in} - R_{1} \cdot i_{in} = v_{in} - v_{R_1} \\
        i_{out} = \frac{1}{R_2} \Big ( -v_{in} + R_1 i_{in} + R_2 i_{in}\Big)
    \end{align*}
$$
</div>

En Python todo este cálculo se puede hacer de forma simbólica con la libraría sympy

1) Se definen los simbolos a utilizar y se generan las matrices de transmision para los elementos del circuito.

    ```python
    import sympy as sp

    R1, R2 = sp.symbols('R1 R2')

    R1_tmatrix = sp.Matrix(
        [[1, R1],
        [0, 1]]
        )

    R2_tmatrix = sp.Matrix(
        [[1, 0],
        [1/R2, 1]]
        )
    ```

2) Se calcula la transmisión total haciendo la multiplicación sucesiva de matrices.

    ```python
    # Total transmission matrix
    total_tmatrix = R1_tmatrix * R2_tmatrix
    display(total_tmatrix)

    # Inverse total transmission matrix
    inverse_total_tmatrix = total_tmatrix.inv()
    display(inverse_total_tmatrix)

    # Voltage gain response
    voltage_gain_response = sp.simplify((1/total_tmatrix[0,0]))
    display(voltage_gain_response)
    ```

3) Multiplicando la matriz de transmisión inversa por las magnitudes de entrada queda:
    
    ```python
    v_in, i_in = sp.symbols('v_in i_in')
    output = inverse_total_tmatrix * sp.Matrix([v_in, i_in])
    ```

## Ejemplo 2: Filtro pasabanda

<p align="center">
  <img src="https://www.researchgate.net/profile/Jonathan-Estevez-Fernandez/publication/312549075/figure/fig3/AS:452414026326019@1484875317721/Figura-5-Filtro-pasa-banda.png" width="70%">
</p>

1) Se definen los simbolos a utilizar y se generan las matrices de transmision para los elementos del circuito.

    ```python
    import sympy as sp

    X_C1, X_C2, R1, R2 = sp.symbols('X_C1 X_C2 R1 R2')

    X_C1_tmatrix = sp.Matrix(
        [[1, X_C1],
        [0, 1]]
    )

    R1_tmatrix = sp.Matrix(
        [[1, 0],
        [1/R1, 1]]
        )

    R2_tmatrix = sp.Matrix(
        [[1, R2],
        [0, 1]]
        )

    X_C2_tmatrix = sp.Matrix(
        [[1, 0],
        [1/X_C2, 1]]
    )
    ```

2) Se calcula la transmisión total haciendo la multiplicación sucesiva de matrices. En este caso, se busca la respuesta en magnitud y fase de la tensión del sistema, por lo que debemos obtener la experesión de la ganancia de tensión a partir de la matriz de transmisión.

    ```python
    # Total transmission matrix
    total_tmatrix = X_C1_tmatrix * R1_tmatrix * R2_tmatrix * X_C2_tmatrix
    display(total_tmatrix)

    # Inverse total transmission matrix
    inverse_total_tmatrix = total_tmatrix.inv()
    display(inverse_total_tmatrix)

    # Voltage gain response
    voltage_gain_response = sp.simplify((1/total_tmatrix[0,0]))
    display(voltage_gain_response)
    ```

3) Para visualizar la respuesta del sistema, primero debemos seleccionar valores para los componentes, en este caso se propone utilizar:

    <div>
    $$
    \begin{align*}
    R_1 &=& 3.1 ~ k\Omega \\
    R_2 &=& 100 ~ k\Omega \\
    C_1 &=& 100 ~ nf \\
    C_2 &=& 1 ~ nf \\
    \end{align*}
    $$
    </div>

    A su vez, se debe seleccionar un rango de frecuencias a representar, y en función de dicho vector de frecuencias, generar las impedancias en función de la frecuencia de cada componente

    ```python
    # Components settings
    R1_value = 3_100
    R2_value = 100_000
    C1_value = 100e-9
    C2_value = 1e-9

    # Frequency settings
    freq_min = 10
    freq_max = 10_000
    freq_bins = 2000

    # Frequency arrays
    freq_array = np.logspace(np.log10(freq_min), np.log10(freq_max), num=freq_bins)
    angular_freq_array = 2 * np.pi * freq_array

    # Impedances as function of frequency
    R1_array = np.ones(freq_bins)*R1_value
    R2_array = np.ones(freq_bins)*R2_value
    X_C1_array = 1/(1j*C1_value*angular_freq_array) 
    X_C2_array = 1/(1j*C2_value*angular_freq_array)
    ```

4) Por último, se debe reemplazar los valores simbolicos de la expresión de ganancia por los vectores de impedancia generados.

    ```python
    # Voltage gain array initialization
    voltage_gain_array = np.zeros(freq_bins, dtype="complex")

    # Loop for replace the symbols by the values of the arrays
    for i in range(freq_bins):
        voltage_gain_i = voltage_gain.subs(
            {
                R1: R1_array[i],
                X_C1: X_C1_array[i],
                X_C2: X_C2_array[i],
                R2: R2_array[i],
                
            }
        )
        voltage_gain_array[i] = voltage_gain_i
    ```

5) Para visualizar la respuesta, se recomienda convertir la respuesta en magnitud del sistema a dB, y el eje de frecuencias configurarlo en escala logarítmica.

    ```python
    from matplotlib import pyplot as plt

    fig, ax1 = plt.subplots(figsize=(10,5))

    # Plot for magnitude
    ax1.plot(freq_array, 20*np.log10(np.abs(voltage_gain_array)), label="Magnitude", color='b')
    ax1.set_xscale('log')
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("Gain [dB]", color='b')
    ax1.set_title("Band-pass response")
    ax1.set_ylim(top=0)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, which="both", ls="--")

    # Plot for phase
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.plot(freq_array, np.angle(voltage_gain_array)*180/np.pi, label="Phase", color=color)
    ax2.set_ylabel("Phase [º]", color=color)
    ax2.set_ylim(top=95, bottom=-95)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yticks(np.arange(-90, 100, 30))
    ax2.set_yticklabels(np.arange(-90, 100, 30))
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')

    fig.tight_layout()
    plt.grid(True, which="both", ls="--")
    plt.show()
    ```

    {{< figure src="images/bandpass-response.png" >}}

## Ejemplo 3: Filtro resonante

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/en/thumb/1/14/RLC_series_band-pass.svg/1280px-RLC_series_band-pass.svg.png" alt="descripcion" width="50%">
</p>



Hasta ahora funciona todo espectacular, pero hay un aspecto no menor de todo el procesamiento anterior que es bastante molesto a la hora de experimentar con la herramienta, y es el tiempo de ejecución y costo computacional.

Como se puede observar en las celdas anteriores, el reemplazo de los valores frecuencia a frecuencia en la expresión simbólica toma mucho tiempo (30 segundos en promedio (es un monton)). 

Esto se debe a que sympy es una muy buena herramienta para trabajar con expresiones matemáticas, pero no está optimizada para el cálculo vectorial. En cambio, trabajar con librerías como NumPy garantiza un mayor rendimiento ya que la mayoría de su código fuente está escrito en lenguajes de bajo nivel como C [(Referencia)](https://stackoverflow.com/questions/45796747/are-sympy-matrices-really-that-slow)

Para el circuito resonante, vamos a encarar la resolución de forma vectorizada con NumPy.

1) Ahora, al trabajar de forma vectorizada, ya no podremos inicializar un objeto simbolico y luego reemplazar valores, si no que la inicialización de las matrices debe ser con los valores de las impedancias frecuencia a frecuencia. Por lo tanto, cambia el orden en el que se definen las variables. Ahora, lo primero que debemos definir es el vector de frecuencia, en conjunto con los bines en frecuencia que queremos representar.

    ```python
    import numpy as np

    freq_min = 10
    freq_max = 100_000
    freq_bins = 2000

    freq_array = np.logspace(np.log10(freq_min), np.log10(freq_max), num=freq_bins)
    angular_freq_array = 2 * np.pi * freq_array
    ```

2) Luego, debemos decidir que valores tendrán los componentes del circuito y generar los vectores de impedancia en función de la frecuencia para cada uno. En este caso se adoptan los siguientes valores:

    <div>
    $$
    \begin{align*}
    R =& 10 ~ \Omega \\
    L =& 1 ~ mH \\
    C =& 1 ~ \mu f \\
    \end{align*}
    $$
    </div>

    ```python
    R_value = 10
    L_value = 1e-3
    C_value = 1e-6

    R_array = np.ones(freq_bins)*R_value
    X_L_array = 1j*angular_freq_array*L_value
    X_C_array = 1/(1j*angular_freq_array*C_value) 
    ```

3) Una vez definidas las impedancias, se conforman las matrices de transmisión de cada elemento. Para operar de forma vectorizada, ahora las matrices serán de:

    <div>
    $$
    (2\times 2 \times \texttt{freq bins})
    $$
    </div>

    ```python
    X_L_tmatrix = np.array(
        [[np.ones(freq_bins), X_L_array],
        [np.zeros(freq_bins), np.ones(freq_bins)]]
    )

    X_C_tmatrix = np.array(
        [[np.ones(freq_bins), X_C_array],
        [np.zeros(freq_bins), np.ones(freq_bins)]]
    )

    R_tmatrix = np.array(
        [[np.ones(freq_bins), np.zeros(freq_bins)],
        [1/R_array, np.ones(freq_bins)]]
    )

    print(R_tmatrix.shape)
    ```

4) Hasta este momento no hay mucha diferencia, pero ahora es cuando se complica (un poco) la cuestión. A la hora de hacer el producto matricial, ya no puedo usar métodos integrados a NumPy porque la multiplicación que necesitamos hacer no es muy común. Recordando que la multiplicación matricial entre dos matrices se puede hacer si la dimensión última dimensión de la primer matriz es igual a la primer dimensión de la segunda matriz, en este caso estamos al horno, porque todas las matrices de transmisión serán de $(2\times 2 \times \texttt{freq bins})$.

    Para solucionar este problema, vamos a crear un producto que lo llamaremos "Layer-Wise Dot Product", ya que estaremos haciendo el producto matricial (Dot product) por capas (Layer-Wise), es decir, vamos a calcular el producto para cada bin de frecuencia.

    ```python
    def layer_wise_dot_product(*matrices: np.ndarray) -> np.ndarray:
        """
        Performs the sequential layer-wise dot product of multiple 3D matrices along the last dimension.

        Args:
        *matrices: A variable number of 3D numpy arrays.

        Returns:
        A 3D numpy array containing the sequential dot product results for each layer in the last dimension.
        """
        if not all(matrix.shape == matrices[0].shape for matrix in matrices):
            raise ValueError("All matrices must have the same dimensions.")

        result = np.zeros_like(matrices[0])

        _, _, num_layers = matrices[0].shape

        for layer_i in range(num_layers):
            # Start the product with the identity matrix for the first multiplication
            dot_product = np.eye(result.shape[0])
            for matrix in matrices:
                dot_product = np.dot(dot_product, matrix[:, :, layer_i])

            result[:, :, layer_i] = dot_product

        return result
    ```

    Ahora si podemos obtener la transferencia haciendo la multiplicación sucesiva de las matrices de transmisión. La desventaja de trabajar de forma vectorizada y no simbolica, es que no tenemos la capacidad de visualizar las ecuaciones en función de los componentes, ahora simplemente observaremos una matriz compleja.

    ```python
    total_tmatrix = layer_wise_dot_product(X_C_tmatrix, X_L_tmatrix, R_tmatrix)

    display(total_tmatrix.shape, total_tmatrix)

    # Output:
    (2, 2, 2000)
    array([[[1.  -1591.54314773j, 1.  -1584.22696366j, 1.  -1576.9444109j ,
         ..., 1.    +62.09489201j, 1.    +62.38313291j,
         1.    +62.67269813j],
        [0. -15915.43147734j, 0. -15842.26963657j, 0. -15769.44410902j,
         ..., 0.   +620.94892013j, 0.   +623.8313291j ,
         0.   +626.72698129j]],

       [[0.1    +0.j        , 0.1    +0.j        , 0.1    +0.j        ,
         ..., 0.1    +0.j        , 0.1    +0.j        ,
         0.1    +0.j        ],
        [1.     +0.j        , 1.     +0.j        , 1.     +0.j        ,
         ..., 1.     +0.j        , 1.     +0.j        ,
         1.     +0.j        ]]])

    ```
5) Para graficar la respuesta de tensión en magnitud y fase del sistema, debemos obtener el array que representa la ganancia de tensión

    ```python
    voltage_gain_array = 1/total_tmatrix[0, 0, :]
    ```

6) Al igual que el ejemplo anterior, se recomienda convertir la respuesta en magnitud del sistema a dB, y el eje de frecuencias configurarlo en escala logarítmica.

    ```python
    from matplotlib import pyplot as plt

    fig, ax1 = plt.subplots(figsize=(10,5))

    # Plot for Gain
    ax1.plot(freq_array, 20*np.log10(np.abs(voltage_gain_array)), label="Magnitude", color='b')
    ax1.set_xscale('log')
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("Gain [dB]", color='b')
    ax1.set_title("Resonant filter response")
    ax1.set_ylim(top=0)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, which="both", ls="--")

    ax2 = ax1.twinx()
    color = 'tab:red'

    ax2.plot(freq_array, np.angle(voltage_gain_array)*180/np.pi, label="Phase", color=color)
    ax2.set_ylabel("Phase [º]", color=color)
    ax2.set_ylim(top=95, bottom=-95)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yticks(np.arange(-90, 100, 30))
    ax2.set_yticklabels(np.arange(-90, 100, 30))
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')

    fig.tight_layout()
    plt.grid(True, which="both", ls="--")
    plt.show()
    ```

    {{< figure src="images/resonant-response.png" >}}