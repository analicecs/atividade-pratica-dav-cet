import numpy as np

def calculate_velocity(keypoints_data, time_step=1):
    """
    Calcula a velocidade dos keypoints (primeira derivada da posição).
    
    Parâmetros:
    keypoints_data -- array numpy com formato (frames, keypoints, dimensions)
    time_step -- intervalo de tempo entre os frames (padrão: 1)
    
    Retorna:
    Um array com as velocidades dos keypoints com formato (frames-1, keypoints, dimensions)
    """
    # Velocidade é a diferença de posições dividida pelo intervalo de tempo

    #TODO: Atividade - Cálculo da velocidade

def calculate_acceleration(keypoints_data, time_step=1):
    """
    Calcula a aceleração dos keypoints (segunda derivada da posição).
    
    Parâmetros:
    keypoints_data -- array numpy com formato (frames, keypoints, dimensions)
    time_step -- intervalo de tempo entre os frames (padrão: 1)
    
    Retorna:
    Um array com as acelerações dos keypoints com formato (frames-2, keypoints, dimensions)
    """
    # Aceleração é a segunda derivada da posição
    # Primeiro calculamos a velocidade

    #TODO: Atividade - Cálculo da aceleração

def calculate_angle(p1, p2, p3):
    """Calcula o ângulo entre três pontos (p1-p2-p3)"""
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Produto escalar e normalização

    #TODO: Calcular produto escalar

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Evitar divisão por zero
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    
    # Calcular ângulo
    
    #TODO: Calcular ângulo
    
    pass
    