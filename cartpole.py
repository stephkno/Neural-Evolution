#!/usr/local/anaconda3/envs/experiments/bin/python3
from matrix_encoding import Network
from copy import copy
import numpy
import gym
import random
import pygame
import math
import sys

pygame.init()
size = (300, 300)
screen = pygame.display.set_mode(size)
pygame.display.flip()

env = gym.make("CartPole-v0")
env._max_episode_steps = 2000

p_states = 1
inputs = 4*p_states
outputs = env.action_space.n

state = env.reset()
state = numpy.array([state for _ in range(p_states)])

num_species = 100
num_tries = 3

agents = []
species = []
winner_agents = []

initial_hidden_node_size = 2
initial_genome_size = 10

render = False
done = False
max_score = 0.0
generation = 0
radius = 10
binary = False

circle_radius = size[0]/2.5
circle_offset = size[0]/2

pygame.display.set_caption("Generation 0 - Fitness 0")

#add gene, remove gene, mutate gene, add node
mutation_probs = numpy.array([10,15,25,0])

mutation_probs = mutation_probs/mutation_probs.sum()

invert = False

def get_new_agent(inputs, outputs):
    agent = Network(inputs, outputs, binary)
    agent.add_node(initial_hidden_node_size)
    agent.add_gene(initial_genome_size)
    agent.add_output_connections()
    return agent

best_agent = get_new_agent(inputs, outputs)

def mutation_curve(x):
    #x = int(numpy.power(2,(x/10))/10)+1 #exponential
    #x = int(numpy.log(x+1)+2)
    return x


def render_network(agent, done, newbest):

    screen.fill((255,255,255))

    pygame.draw.rect(screen, (0, 0, 0), [0, 0, 10, int(10 * agent.softmax(agent.nodes)[-1])], 0)
    pygame.draw.rect(screen, (0, 0, 0), [10, 0, 10, int(10 * agent.softmax(agent.nodes)[-2])], 0)

    step = math.pi * 2 / agent.size
    node_pos = []

    for i in range(len(agent.nodes)):
        x = math.cos(step * i) * circle_radius + circle_offset
        y = math.sin(step * i) * circle_radius + circle_offset
        node_pos.append([x, y])

    r = int(radius / (len(agent.nodes)/20))

    #draw links
    for i, row in enumerate(agent.genome):
        for z, link in enumerate(row):

            #if invert:
            #    col = [255-(255*invert),255-(255*invert),255-(255*invert)]
            #else:
            #    col = [0,0,0]

            #if link > 0.0:
            #    col[1] = int((255 * (link+1)/2))
            #else:
            #    col[0] = int(255 - (255 * (link+1)/2))
            col = [0,0,0]
            if link != 0.0 and i != z:
                col[0] = 255*((link+1)/2)

                pygame.draw.line(screen, col, [int(node_pos[i][0]+r/4), int(node_pos[i][1]+r/4)], [int(node_pos[z][0]+r/4), int(node_pos[z][1]+r/4)], 1)
            elif i == z and link != 0.0:
                pygame.draw.circle(screen, (0, 0, 0), [int(node_pos[i][0]), int(node_pos[i][1])], 2)

    output_counter = 0

    #draw nodes
    for i, pos in enumerate(node_pos):

        x, y = pos[0], pos[1]

        if i < inputs:
            color = [255, 200, 45] #input color - yellow
        elif i >= agent.size - outputs:
            color = [252, 102, 32] #output color - orange
        else:
            color = [106, 202, 37] #hidden color - green

        activation_value = (agent.activation(agent.nodes[i]) + 1) / 2

        if i >= agent.size - agent.outputs:
            output = agent.nodes[(-outputs):]
            action = numpy.argmax(output)
            activation_value = (action == output_counter)*1.0
            output_counter+=1

        if render:
            color[0] = int(color[0] * activation_value)
            color[1] = int(color[1] * activation_value)
            color[2] = int(color[2] * activation_value)

        pygame.draw.circle(screen, color, (int(x), int(y)), r, r)

        # draw links
    for i, row in enumerate(agent.genome):
        for z, link in enumerate(row):
            if i == z and link != 0.0:
                pygame.draw.circle(screen, (255, 255, 255), [int(node_pos[i][0]), int(node_pos[i][1])], int(radius/4))

    pygame.display.flip()

def sex(agent1, agent2):

    genome1 = agent1.genome
    genome2 = agent2.genome

    mask1 = ((numpy.random.random((agent1.size, agent1.size)) > 0.5)*2.0)-1
    mask2 = mask1 * -1.0
    mask1 = (mask1 > 0.0)*1.0
    mask2 = (mask2 > 0.0)*1.0

    genome1 = genome1 * mask1
    genome2 = genome2 * mask2
    new_genome = genome1+genome2

    new_agent = Network(inputs, outputs, binary)
    new_agent.add_node(initial_hidden_node_size)

    new_agent.genome = new_genome
    return new_agent

def speciate(agents):

    new_agents = []

    for i in range(int(num_species-len(new_agents))):
        baby = sex(agents[-1], agents[-2])

        if numpy.random.random() > 0.5:
            baby.add_gene(mutation_curve(i))
        new_agents.append(baby)

    return new_agents

#get first generation of agents
agents = [get_new_agent(inputs, outputs),get_new_agent(inputs, outputs)]
agents = speciate(agents)

def get_key(element):
    return element.fitness

while True:# max_score < 2000:

    for i,agent in enumerate(agents):

        score = 0.0
        best_of_gen = 0.0

        for _ in range(num_tries):
            steps = 0
            while not done:

                for event in pygame.event.get():  # User did something
                    if event.type == pygame.QUIT:  # If user clicked close
                        break
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            render = not render
                            print("Render: {}".format(render))


                x = agent.eval(state)

                action = int(numpy.argmax(x))
                #action = numpy.random.choice(range(outputs), p=x)

                new_state, reward, done, _ = env.step(action)

                if render:
                    render_network(agent, done, (steps > max_score))

                if render:
                    pygame.display.set_caption(
                    "[({})|Generation {}|Max{}|Species#{}]".format(steps, generation, round(max_score), i))

                steps += 1

                score += reward
                state = numpy.roll(state, -1, axis=0)
                state[-1] = new_state

                if render and steps % 2 == 0:
                    env.render()

           # print("Steps:{}".format(steps))
            agent.fitness = score
            done = False
            state = env.reset()
            state = numpy.array([state for _ in range(p_states)])

        if score > max_score:
            max_score = score
            best_agent = copy(agent)
            if not render and steps % 2 == 0:
                render_network(best_agent, done, True)
            print(".",end="")
            pygame.display.set_caption(
                "[({})|Generation {}|Max{}|Species#{}]".format(steps, generation, round(max_score), i))

    #if generation > 25:
    #    run_episode(best_agent)

    state = env.reset()
    state = numpy.array([state for _ in range(p_states)])

    agents = sorted(agents, key=get_key, reverse=False)
    #for agent in agents:
    #    print(agent.fitness)

    print("[Generation:#{}|Max Fitness:{}]".format(generation,max_score),end="\r")
    #pygame.display.set_caption("Generation {} - Fitness {}".format(generation, max_score))
    winner_agents = agents[int(len(agents)/2):]
    agents = speciate(winner_agents)
    winner_agents = []

    generation += 1
    sys.stdout.flush()

#endless loop - for victory!
