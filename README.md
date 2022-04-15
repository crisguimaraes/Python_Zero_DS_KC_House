# Python_Zero_DS_KC_House
## 1. Objetivo
Este projeto de insights foi criado para encontrar as melhores oportunidades de compra e venda de imóveis para a House Rocket, uma imobiliária fictícia, localizada no condado de King nos EUA, incluindo a região de Seattle.

O modelo de negócio da House Rocket consiste em uma plataforma digital para a compra e venda de imóveis online.

O dataset utilizado contém informações de 21613 imóveis comercializados nesta região entre maio de 2014 e maio de 2015 e 20 atributos conforme indicado na tabela abaixo. Cada imóvel é identificado por um ‘id’.

Atributo | Definição
------------ | -------------
|id|	Identificação única do imóvel.|
|date|	Data de venda.|
|price|	Preço de venda.|
|bedrooms|	Número de quartos.|
|bathrooms|	Número de banheiros.|
|sqft_living|	Tamanho do espaço interno dos imóveis em pés quadrados.|
|sqft_lot|	Tamanho do terreno em pés quadrados.|
|floors|	Número de andares.|
|waterfront|	‘1’ se o imóvel tem vista para a água, ‘0’ se não.|
|view|	Índice de 0 a 4 indicando o quão boa é a vista do imóvel.|
|condition|	Condição do imóvel ranqueado de 0 a 5.|
|grade|	Classificação pela qualidade da construção. Construções de melhor qualidade custam mais caro para construir por unidade de medida.|
|sqft_above| Tamanho do imóvel acima do nível do solo em pés quadrados.|
|sqft_sqft_basement|	Tamanho do porão em pés quadrados.|
|yr_built|	Ano de construção do imóvel.|
|yr_renovated|	Ano no qual o imóvel foi renovado. 0 indica que o imóvel não passou por reforma.|
|zipcode|	Código de 5 dígitos para indicar a região onde se encontra o imóvel.|
|lat|	Latitude.|
|long|	Longitude.|
|sqft_living15|	Tamanho médio do espaço interno dos 15 imóveis mais próximos em pés quadrados.|
|sqft_lot15|	Tamanho médio dos terrenos dos 15 imóveis mais próximos em pés quadrados.|

## 2. Problema de negócio
Por possuir um portfólio de imóveis grande e variado, o time de vendas da House Rocket tem encontrado dificuldades em analisar o conjunto de dados da forma tradicional e têm despendido muito tempo e esforços na escolha das melhores oportunidades de negócios.

## 3. Premissas do negócio
As sugestões de compra e venda foram feitas assumindo-se que todas as casas disponíveis no portfólio estão em boas condições;

Foi utilizado o critério de sazonalidade, considerando verão e inverno na análise dos dados;

Necessidade de a análise ser disponibilizada para o time de vendas de forma online e através de dispositivos móveis.

## 4. Demandas do negócio

•	Dashboard interativo com todas as informações sobre o portfólio de imóveis da House Rocket e com possibilidade de o usuário fazer suas próprias análises através de filtros para:
1. separação dos imóveis por código postal;
2. seleção de atributos específicos;
3. visualização das métricas descritivas definidas para os atributos;
4. visualização de um mapa com a densidade de portfólio por região;
5. visualização de um mapa com a densidade de preços por região;
6. checagem da variação anual dos preços dos imóveis;
7. checagem da variação diária dos preços dos imóveis;
8. checagem da distribuição dos imóveis por: preço, número de quartos, número de andares, vista para água ou não.
    
•	Respostas para as questões:
1. Quais os imóveis a House Rocket deveria comprar e por qual preço?
2. Uma vez que o imóvel seja comprado, qual o melhor momento para vendê-lo e por qual preço?

## 5. Planejamento da solução

### 5.1 Estratégia para a resolução do problema
•	Discussão com o time de negócios para o entendimento do problema;

•	Coleta dos dados;

•	Limpeza dos dados;

•	Análise exploratório dos dados;

•	Levantamento de hipóteses;

•	Validação das hipóteses;

•	Criação de um dashboard;

•	Disponibilização do dashboard em produção.

### 5.2	Ferramentas
•	Python 3.9.7;

•	PyCharm;

•	Jupyter Notebook.

### 5.3	Produto final
Dashboard interativo que possa ser acessado de forma remota e contendo as análises solicitadas pelo time.

### 5.4	Detalhamento da solução
1. Tabela contendo todos os imóveis do portfólio com seus respectivos atributos. 
Filtro que permita selecionar os imóveis por código postal e por atributos específicos;

2. Tabela contendo o número de imóveis e a média de preços por código postal. 
Filtro que permita selecionar um ou mais códigos;

3. Tabela contendo a estatística descritiva (valor mínimo, máximo, médio, mediano, e desvio padrão) dos atributos dos imóveis.
    
4. Mapa contendo a densidade de imóveis por código postal;

5. Mapa contendo a densidade de preços por código postal;

6. Gráfico contendo a variação do preço médio por ano de construção do imóvel;

7. Gráfico contendo a variação do preço médio por dia;  

8. Gráfico de distribuição de preços dos imóveis;

9. Gráfico de distribuição do número de quartos, banheiros, andares e vista para a água;

10. Todos os gráficos contendo filtros para interatividade do usuário;

11. Relatório de imóveis recomendados para compra, que estejam abaixo do preço mediano da sua região e em boas condições;

12. Mapa exibindo a localização dos imóveis indicados para compra;

13. Relatório contendo a melhor estação para venda imóveis e valor de venda recomendado. 
• Para cada região com imóveis sugeridos para compra, identificar a variação de preço médio por sazonalidade, para sugerir a venda na sazonalidade onde os valores são mais elevados.

• Se o preço da compra do imóvel for maior que o preço médio da região + sazonalidade de maior preço, o preço da venda sugerido será igual ao preço da compra + 10%. Se o preço da compra do imóvel for menor que o preço médio da região + sazonalidade de maior preço, o preço da venda sugerido será igual ao preço da compra + 30%.

## 6. Hipóteses

### H1 – Imóveis que possuem vista para a água são 30% mais caros, na média.

A média de preços dos imóveis com vista para a água ($1,661,876.02) é 212% maior que a média de preços dos imóveis sem vista para a água ($531,563.60). 

### H2 – Imóveis com data de construção menor que 1955 são 50% mais baratos na média.
A média dos preços dos imóveis construídos antes de 1955 ($563,940.14) é apenas 4,4% maior que a média de preços dos imóveis construídos após essa data ($540,213.15).

### H3 – Imóveis com porão são 50% maiores do que imóveis sem porão.
A média de tamanho do interior das casas sem porão (2362.44 sqft) é 46% maior do que a média de tamanho das casas com porão (3451.45 sqft).

### H4 – O crescimento médio do preço dos imóveis por ano de construção é de 10%.
O crescimento médio dos preços dos imóveis por ano de construção é de 1%.

## 7. Conclusão
O objetivo inicial de obter as melhores oportunidades de compra e venda dos imóveis da House Rocket foi alcançado. A solução foi disponibilizada de forma online e interativa para o time de vendas que agora tem autonomia para realizar suas próprias análises pela internet e através de um app.

## 8. Observações
• O dashboard com o projeto em produção pode ser acessado aqui [Heroku](http://house-rocket-cristiane.herokuapp.com/)

•	Este projeto é parte do curso "Python do Zero ao DS", da [Comunidade DS](https://www.comunidadedatascience.com/)

•	Os dados utilizados são públicos e foram obtido em [Kaglle](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
