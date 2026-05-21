def perceptron(x, w, alpha, esperado):
   soma = 0
   for i in range(len(x)):
        soma += x[i] * w[i]
      
        if soma >= 0:
            saida = 1
        else:
            saida = 0

        erro = esperado - saida

        for i in range(len(w)):
            w[i] = w[i] + alpha * erro * x[i]

        return saida, erro, w
       
x = [1, 0, 1]

w = [0.2, -0.1, 0.4]

alpha = 0.1

esperado = 1

saida, erro, novos_pesos = perceptron(x, w, alpha, esperado)
print("Saída: ", saida)
print("Erro: ", erro)
print("Novos pesos: ", novos_pesos)


x = [1, 0, 1]

w = [0.2, -0.1, 0.4]

alpha = 0.1

esperado = 0

saida, erro, novos_pesos = perceptron(x, w, alpha, esperado)
print("Saída: ", saida)
print("Erro: ", erro)
print("Novos pesos: ", novos_pesos)