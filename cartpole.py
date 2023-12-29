from matrix import Network
from matrix import Utils
from copy import copy
import numpy
import gym
import random
#import pygame
import math
import sys
import skimage.measure
import cv2
import atexit
from threading import Thread

def preprocess(x):
    return numpy.array(x).reshape(-1,1)/10

util = Utils()
train = True

#pygame.init()
size = (300, 300)
#screen = pygame.display.set_mode(size)
#pygame.display.flip()
width, height = 10,10

env = gym.make("CartPole-v1")
win_state = 2000
env._max_episode_steps = 2000

if len(sys.argv) > 1:
    filename = sys.argv[1]
    best_agent = util.import_genome(filename)
    train = False
else:
    print("New Agent")

p_states = 3
inputs = 4*p_states
outputs = 2

state = env.reset()
print(state)
state = preprocess(state[0])
state = numpy.array([state for _ in range(p_states)])

n_agents = 25
n_init_agents = 25
num_species = 0
num_tries = 1

agents = []
species = []
winner_agents = []

initial_hidden_node_size = 16
initial_genome_size = 100

render = False
done = False
max_score = 0.0
generation = 0
radius = 10
species_count = 0
lc = 0

binary = True

circle_radius = size[0]/2.5
circle_offset = size[0]/2

#pygame.display.set_caption("")

#add gene, remove gene, mutate gene, add node
mutation_probs = numpy.array([10,15,25,0])
mutation_probs = mutation_probs/mutation_probs.sum()
invert = False

def get_new_agent(inputs, outputs):
    agent = Network(inputs, outputs, binary, species_count)
    agent.add_nodes(initial_hidden_node_size)
    agent.add_gene(initial_genome_size)
    return agent


def mutation_curve(x):
    #x = int(numpy.power(2,(x/10))/10)+1 #exponential
    #x = int(numpy.log(x+1)+2)
    return x

def render_network(agent, done, newbest, steps):
    pygame.display.set_caption(
        "[({})|Generation {}|Max{}|Species#{}]".format(steps, generation, round(max_score), species_count))

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
        if agent.render:
            #if i >= agent.size - agent.outputs:
            #    output = agent.nodes[(-outputs):]
            #    action = numpy.argmax(output)
            #    color[0] = (action == output_counter)*color[0]
            #    color[1] = (action == output_counter)*color[1]
            #    color[2] = (action == output_counter)*color[2]
#
#                output_counter+=1
#
#            else:
            color[0] = (activation_value > 0.5)*color[0]
            color[1] = (activation_value > 0.5)*color[1]
            color[2] = (activation_value > 0.5)*color[2]

        pygame.draw.circle(screen, color, (int(x), int(y)), r, r)

        # draw links
    for i, row in enumerate(agent.genome):
        for z, link in enumerate(row):
            if i == z and link != 0.0:
                pygame.draw.circle(screen, (255, 255, 255), [int(node_pos[i][0]), int(node_pos[i][1])], int(radius/4))

    pygame.display.flip()

#get first generation of agents
agents = [get_new_agent(inputs, outputs) for _ in range(n_init_agents)]

c = 0
for a in agents:
    a.id += c
    c+=1
num_species = c

def get_key(element):
    return element.fitness

def render_env(state):

    #state = numpy.transpose(state, (1, 0))
    state = cv2.resize(state, (10*width, 10*height), interpolation=cv2.INTER_AREA)
 
    cv2.imshow("state", state)
    cv2.waitKey(1)


def run_episode(agent):
    env = gym.make("CartPole-v1", render_mode="human")
    env._max_episode_steps = 999999

    done = False
    new_state, _ = env.reset()
    
    state = env.reset()
    state = preprocess(state[0])
    state = numpy.array([state for _ in range(p_states)])

    steps = 0
    score = 0

    while not done:
#        for event in pygame.event.get():  # User did something
#            if event.type == pygame.QUIT:  # If user clicked close
#                break

        x = agent.evaluate(state)
        env.render()

        action = int(numpy.argmax(x))
        #action = numpy.random.choice(range(outputs), p=x/x.sum())
        new_state, reward, done, _, x = env.step(int(action))

        new_state = preprocess(new_state)
        steps += 1
        score += reward
        env.render()

        #if agent.render:
            #render_network(agent, done, (steps > max_score), steps)
            #render_env()

        state = numpy.roll(state, -1, axis=0)
        state[-1] = new_state

    done = False


def train_episode(agents, species_count, num_species, max_score, generation):

    env = gym.make("CartPole-v1")

    done = False
    state = env.reset()
    state = preprocess(state[0])
    state = numpy.array([state for _ in range(p_states)])

    for i,agent in enumerate(agents):
        #agent.render = False
        #print("[Agent#{}]".format(species_count), end="\r")
        sys.stdout.flush()
        species_count += 1
        score = 0.0


        for _ in range(num_tries):
            steps = 0
            while not done:

                for event in []:#pygame.event.get():  # User did something
                    if event.type == pygame.QUIT:  # If user clicked close
                        break
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            agent.render = not agent.render
                        elif event.key == pygame.K_q:
                            train = False
                            break

                x = agent.evaluate(state)

                action = int(numpy.argmax(x))
                #action = numpy.random.choice(range(outputs), p=x/x.sum())
                new_state, reward, done, _, x = env.step(int(action))

                new_state = preprocess(new_state)
                steps += 1
                score += reward

                #if agent.render:
                    #render_network(agent, done, (steps > max_score), steps)
                #env.render()

                state = numpy.roll(state, -1, axis=0)
                state[-1] = new_state

                if steps > win_state:
                    done = True
                    return 

                #if render and steps % 2 == 0:
                   # env.render()

           # print("Steps:{}".format(steps))

            done = False
            state = env.reset()

            state = preprocess(state[0])
            state = numpy.array([state for _ in range(p_states)])

        agent.set_fitness(score)

        score = 0

        if generation % 10 == 0 and agent.fitness >= win_state:
            train = False
            break


best_agent = get_new_agent(inputs, outputs)
best_agent_id = 0

#    thread = Thread(target = run_episode, args = (best_agent, ))
#    thread.start()

while train:
    train_episode(agents, species_count, num_species, max_score, generation)

    #if generation > 25:
    #    run_episode(best_agent)

    #generation is finished
    state = env.reset()
    state = preprocess(state[0])

    state = numpy.array([state for _ in range(p_states)])
    agents = sorted(agents, key=get_key, reverse=False)

    for i,a in enumerate(agents):
        print("|Agent {}|Fitness: {}|".format(a.id, a.fitness))
    

    print("\n[Generation:#{}|Max Fitness:{}|Size:{}|BestAgent:{}]".format(generation, max_score, best_agent.genome.size, best_agent.id))
    agents, num_species = util.generate_agents(agents[int(len(agents)/2):], best_agent, num_species, inputs, outputs, binary, initial_hidden_node_size, n_agents)
    winner_agents = []

    sys.stdout.flush()
    generation += 1

    for agent in agents:
        if agent.fitness > max_score:
            max_score = agent.fitness
            best_agent = copy(agent)
            best_agent_id = species_count


print("\nEvaluation...")
render = True
agent.render = True

while True:
    run_episode(best_agent)