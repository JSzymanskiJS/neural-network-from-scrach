import numpy as np

#Sieć będzie miała 1404 wejścia, jedną warstwę ukrytą o 500 neuronach i 5 wyjść

class Dense_Layer:    
    def __init__(self, n_inputs, n_neurons, learning_coefficient: float = 0.2) -> None:
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # Wagi po "lewej" stronie warstwy.
        self.biases = np.zeros((1, n_neurons)) # Wartości progowe
        self.is_summed = False # Flaga propagacji w przód 
        self.learning_coefficient = learning_coefficient # Współczynnik uczenia odpowiada na pytanie: "Jak bardzo mocno ma się poprawiać algorytm za każdą sesją uczenia się?".
    
    def sigmoid(self, inputs):
        self.output = 1/(1 + np.exp(-inputs))
    # def sigmoid_derivative(self, outputs_backward): 
    #     self.sigmoid_backward_output = outputs_backward * (1 - outputs_backward)
    
    def forward_propagation(self, inputs):
        self.sums = np.dot(inputs, self.weights) + self.biases
        self.is_summed = True
        return self.sigmoid(self.sums)
    def backward_propagation(self, desired_outputs, inputs):
        if(self.is_summed):
            # Obliczanie kosztu
            cost = (desired_outputs - self.activation) * (self.activation * (1-self.activation)) #delta
            #zmiany wag aktualnej warstwy 
            self.weights = self.weights + self.learning_coefficient * cost * inputs #TODO Uogólnij to dla każdej możliwej warstwy
            
            self.is_summed = False
            return cost
        

if __name__ == "__main__":
    layer1 = Dense_Layer(1404, 500)
    layer2 = Dense_Layer(500, 5)
    
    # Uczenie się
    # Propagacja w przód
    
    # Propagacja wstecz
    