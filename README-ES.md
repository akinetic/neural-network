# Red Neuronal

> **Este proyecto presenta una manera alternativa y totalmente distinta de entrenar una red neuronal y una manera muy simplificada de procesar los datos luego de su entrenamiento.**

---

## I. DataFrame

Consideremos que un DataFrame (Purificado) es una Base de Datos acotado no continuo donde cada entrada está asociada con una única salida, por lo tanto, no habiendo dos salidas para una misma entrada. Por otro lado, no habiando tampoco entrada-salida iguales repetidos (es decir, un DataFrame sin Densidad o Profundidad)

En este proyecto, nuestro Dataframe (Purificado) o modelo de juguete es el siguiente:

[-6.00,-6.00]

[+2.00,+4.00]

[-8.00,-4.00]

[+0.00,+0.00]

[+4.00,+10.0]

[-4.00,-6.00]

[+6.00,+18.0]

[-5.00,-6.01]

[+3.00,+7.00]

[-2.00,-4.00]

Donde el primer valor (lado izquierdo) representa la entrada y el segundo valor (lado derecho) representa la salida asociada a la entrada más próxima del lado izquierdo.

---

## II. Creación del Diccionario Base

La creación del Diccionario Base simplemente consiste en ordenar de menor a mayor los Datos del DataFrame.

En nuestro modelo de juguete, Diccionario Base es el siguiente:

[-8.00,-4.00]

[-6.00,-6.00]

[-5.00,-6.01]

[-4.00,-6.00]

[-2.00,-4.00]

[+0.00,+0.00]

[+2.00,+4.00]

[+3.00,+7.00]

[+4.00,+10.0]

[+6.00,+18.0]

El Diccionario Base ya es funcional a pesar de que aún no está optimizado ni comprimido.

Por ejemplo, si la entrada X coincide con un valor Xn del Diccionario Base entonces el Procedimiento consiste en que la salida es el valor Y asociado a Xn.

Sin embargo, si la entrada X no coincide con un valor Xn del Diccionario Base y además (por el momento) X es mayor que el valor menor Xn y mayor que el valor Xm del Diccionario Base entonces el Procedimiento consiste en que la salida Y, está dada por la siguiente ecuación:

Y = (X - X1) . [ (Y2 - Y1) / (X2 - X1) ] + Y1

Donde X1 el valor más próximo menor a X en el Diccionario Base, X2 es el valor más próximo mayor a X en el Diccionario Base y Y2 y Y1 son los valores asociados a X2 y X1 respectivamente en el Diccionario Base.

Por ejemplo, en nuestro modelo de juguete, si la entrada X es 5 entonces la salida Y, está dada por:

Y = (5 - 4) . [ (18 - 10) / (6 - 4) ] + 10

Y = 14

---

## III. Optimización del Diccionario Base

La optimización del Diccinario Base consiste en hacer más funcional al Diccionario Base bajo un determinado criterio o varios.

El criterio más simple elegido por este proyecto consite en primer lugar en obtener las Pendientes(Pesos) y las Ordenadas(Sesgos) de cada par más próximo de los datos del Diccionario Base.

En nuestro ejemplo, resulta:

[-8.00,-4.00]

				(-1.00,-12.0)
				
[-6.00,-6.00]

				(-0.01,-6.06)
				
[-5.00,-6.01]

				(+0.01,-5.96)
				
[-4.00,-6.00]

				(+1.00,-2.00)
				
[-2.00,-4.00]

				(+2.00,+0.00)
				
[+0.00,+0.00]

				(+2.00,+0.00)
				
[+2.00,+4.00]

				(+3.00,-2.00)
				
[+3.00,+7.00]

				(+3.00,-2.00)
				
[+4.00,+10.0]

				(+4.00,-6.00)
				
[+6.00,+18.0]

Posteriormente, se asocia lo obtenido con el dato más próximo anterior y a cada dato del Diccionario Base se le quita su valor Y, quedando en nuestro ejemplo de la siguiente manera:

[-8.00] (-1.00,-12.0)

[-6.00] (-0.01,-6.06)

[-5.00] (+0.01,-5.96)

[-4.00] (+1.00,-2.00)

[-2.00] (+2.00,+0.00)

[+0.00] (+2.00,+0.00)

[+2.00] (+3.00,-2.00)

[+3.00] (+3.00,-2.00)

[+4.00] (+4.00,-6.00)

[+6.00] (-----,-----)

Este nuevo Diccionario Optimizado como se verá luego es mucho más funcional que el Diccionario Base y ya es funcional también como lo es el Diccinario Base para poder ser utilizado.

Por ejemplo, si la entrada X es mayor o igual que el valor menor Xn y mayor o igual que el valor Xm del Diccionario Optmizado (por el momento) entonces el Procedimiento consiste en que la salida Y, está dada por la siguiente ecuación:

Y = X . Pendiente(Peso) + Ordenada(Sesgo)

Donde la Pendiente(Peso) y la Ordenada(Sesgo) son la Pendiente(Peso) y la Ordenada(Sesgo) asociadas al Xn más próximo menor o igual del Diccionario Optimizado.

Por ejemplo, en nuestro modelo de juguete, si la entrada X es 5 entonces la salida Y, está dada por:

Y = 5 . 4 - 6

Y = 14

---

## IV. Compresión sin Perdida de Información

La Compresión sin Perdida de Información consiste en eliminar Datos o Neuronas redundantes del Diccionario Optimizado.

Básicamente consiste en eliminar el Dato o la Neurona con el valor más próximo mayor que tiene igual Pendiente al Dato o Neurona más próximo menor anterior, dado que ese Dato o Neurona redundante está implícitamente contenido por el Dato o Neurona más proximo menor anterior.

En nuestro modelo de juguete, los Datos o las Neuronas Redundantes son: [+0.00] (+2.00,+0.00) y [+3.00] (+3.00,-2.00), quedando, por lo tanto, el Diccionario Optimizado de nuestro modelo de juquete y sin perdida alguna de información de la siguiente manera:

[-8.00] (-1.00,-12.0)

[-6.00] (-0.01,-6.06)

[-5.00] (+0.01,-5.96)

[-4.00] (+1.00,-2.00)

[-2.00] (+2.00,+0.00)

[+2.00] (+3.00,-2.00)

[+4.00] (+4.00,-6.00)

[+6.00] (-----,-----)

---

## V. Compresión con Perdida de Información no Relevante

La Compresión con Perdida de Información no Relevante consiste en eliminar Datos o Neuronas no relevantes del Diccionario Optimizado.

Básicamente consiste en eliminar el Dato o la Neurona cuya eliminación no produce localmente un cambio relevante en el resultado de salida.

Cuando un Dato o una Neurona no es relevante localmente depende de la exactitud que se quiere obtener de la Red Neuronal

En nuestro modelo de juguete se ha determinado que una diferencia menor a 0.03 entre el DataFrame y el valor de salida dado por el Diccionario Optimizado y Comprimido es aceptable.

Por lo tanto, en nuestro modelo de juguete, el Dato o Neurona Redundantes son: [-5.00] (+0.01,-5.96), quedando, por lo tanto, el Diccionario Optimizado de nuestro modelo de juquete y con perdida de información no relevante de la siguiente manera:

[-8.00] (-1.00,-12.0)

[-6.00] (+0.00,-6.00)

[-4.00] (+1.00,-2.00)

[-2.00] (+2.00,+0.00)

[+2.00] (+3.00,-2.00)

[+4.00] (+4.00,-6.00)

[+6.00] (-----,-----)

Ahora en el Diccionario Optimizado y Comprimido de nuestro modelo de juguete para X igual a -5 el resultado es -6, obteniendose solamente una diferencia de 0.01 con respecto al dato real del DataFrame o Diccionario Base (-6.01)

---

## VI. Otras Compresiones sin Perdida de Información y/o con Perdida de Información no Significativa

Funciones Globales y sobre todo Locales (Ajuste en el Diccionario Comprimido y en el Procedimiento )

---

## VII. Más allá del Extremo Menor y Mayor

---

## VIII. Observaciones Generales

+ La Red Neuronal puede analizar y modifical el Diccionario Optimizado sin tener en cuenta el resto del Diccionario y sin producir tampoco cambios al resto del Diccionario.

+ Se podría visualizar como que cada Neurona se ocupa de un sector de la Red y que su funcionamiento no altera la Red más allá de ese sector de la Red.

+ Además entre las Neuronas locales (más próximas entre sí) se puede asocian y/o unificar para hacer de la Red Neuronal más compacta y eficiente

+ Lo ideal sería que el Diccionario Optimizado sea lo más compacto posible, que la velocidad de entrada/salida sea lo más rápida posible y con el menor costo posible, y finalmente que la Red Neuronal pueda evolucionar de la manera más flexible/sencilla posible.

---

## IX. Bibliografía

---

Alex Kinetic

LICENCIA MIT
