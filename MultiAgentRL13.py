import numpy as np
import gym
from gym import spaces
import pygame
import math
import matplotlib.pyplot as plt
import pandas as pd
import random
import colorsys
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from collections import Counter
import pygame
from scipy.spatial.distance import cdist, pdist, squareform

import cProfile
import colorsys
import webcolors

ENV_SIZE = 5
ENERGY_TO_SIZE_FACTOR = 100
MIN_AGENT_ENERGY = 10
ENERGY_DECAY_RATE = 0.999
AGENT_SPEED_FACTOR = 0.025
SIGHT_RADIUS = 1#0.5  # Named constant for sight radius
pan_speed = 50
max_turn_angle = 15
max_sprint_duration = 20  # Maximum sprint duration in seconds
SPRINT_REST_DURATION = 40 # Time to regain stamina in seconds
sprint_rest_duration = SPRINT_REST_DURATION
sprint_speed_multiplier = 1.5  # Sprint speed multiplier
mutation_radius = 0.1
SECONDS = 10
MINUTES = 60 * SECONDS
BABY_TIME = 2 * SECONDS


NUM_SPECIES = 1
MIN_POPULATION = 20
MAX_TOTAL_POPULATION = 99
MAX_SPECIES_POP = 60
LIFESPAN = 3 * MINUTES

SAVE_INTERVAL = 1 * MINUTES
INPUT_SIZE = 9
HIDDEN_SIZE = 6
OUTPUT_SIZE = 3
FOOD_PER_SECOND = 20
MAX_FOOD = 200

AGENT_IMAGE = pygame.image.load('agent_bw.png')
FACE_MOUTH_OPEN_IMAGE = pygame.image.load('face_mouth_open.png')
FACE_MOUTH_CLOSED_IMAGE = pygame.image.load('face_mouth_closed.png')
FOOD_IMAGE = pygame.image.load('food.png')


food_distance_cache = {}
agent_distance_cache = {}

t = 0
class Food:
    def __init__(self, x, y,energy, image=FOOD_IMAGE, direction = None):
        self.x = x
        self.y = y
        self.energy = energy
        self.speed = 0
        self.image = image
        self.direction = direction if direction is not None else np.random.uniform(-np.pi, np.pi)
        self.body_size = math.sqrt(self.energy) / ENERGY_TO_SIZE_FACTOR
        self.ancestor = None

    def can_eat(self, other):
        return False


class Agent:



            

    def __init__(self, x, y, direction, speed, baby_energy, mature_energy, species_id, hue, father=None, mother=None,  generation=0):
        



        # Initialize position, direction, speed and energy attributes
        self.x, self.y = x, y
        self.direction = direction
        self.speed = speed
        self.energy = baby_energy
        self.body_size = self.calculate_body_size()
        self.mouth_open = False
        
        # Initialize behavior attributes
        self.sprint = False
        
        # Initialize inheritance attributes
        self.ancestor = species_id
        self.generation = generation
        self.hue =  hue
        
        # Initialize reproductive and sprint attributes
        self.sprint_duration = 0
        self.sprint_rest_timer = SPRINT_REST_DURATION
        self.baby_rest_timer = 0
        self.children = []
        self.age = 0

        # Initialize growth attributes
        self.baby_energy = max(1.5 * MIN_AGENT_ENERGY, baby_energy)
        self.mature_energy = mature_energy


        if father is not None and mother is not None:
            self.father = father
            self.mother = mother

            self.father_input_weights, self.father_output_weights = father.get_meosis_genes()
            self.mother_input_weights, self.mother_output_weights = mother.get_meosis_genes()

            self.mutate()

        else:

                        
            self.father_input_weights = np.random.uniform(-1.0, 1.0, size=(INPUT_SIZE+1,HIDDEN_SIZE))
            self.mother_input_weights = np.random.uniform(-1.0, 1.0, size=(INPUT_SIZE+1,HIDDEN_SIZE))


            self.father_output_weights = np.random.uniform(-1.0, 1.0, size=(HIDDEN_SIZE+1, OUTPUT_SIZE))
            self.mother_output_weights = np.random.uniform(-1.0, 1.0, size=(HIDDEN_SIZE+1, OUTPUT_SIZE))

        self.input_weights = ( self.father_input_weights + self.mother_input_weights ) / 2.0
        self.output_weights = ( self.father_output_weights + self.mother_output_weights ) / 2.0

        self.set_image_tint()

    def set_image_tint(self):

        # Get the RGB color of the tint based on the agent's hue
        tint_color = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(self.hue, 0.5, 0.8))


        # Create a copy of the agent image to apply the tint
        self.image = AGENT_IMAGE.copy()

        # Apply the tint color
        self.image.fill(tint_color, special_flags=pygame.BLEND_RGBA_MULT)

    # Define the mutation function
    def mutate_np(self,  x):
        return x + np.random.normal(0, 0.1)

    def mutation_percent(self):
        return (1 + 2 * random.random()*mutation_radius - mutation_radius)

    def mutation(self):
        return (2.0 * random.random()*mutation_radius - mutation_radius)

    def mutate(self):
        self.baby_energy *= self.mutation_percent()
        self.mature_energy *= self.mutation_percent()
        self.mature_energy = max(2 * self.baby_energy, self.mature_energy)

    def adjust_within_bounds(self, attribute, upper_bound=1):
        return min(max(attribute, 0), upper_bound)

    def get_meosis_genes(self):

        # Random condition array with the same shape
        condition = np.random.choice([True, False], size=self.father_input_weights.shape)

        # Create the output array
        input_genes = np.where(condition, self.father_input_weights, self.mother_input_weights)

        # Random condition array with the same shape
        condition = np.random.choice([True, False], size=self.father_output_weights.shape)

        # Create the output array
        output_genes = np.where(condition, self.father_output_weights, self.mother_output_weights)

        # Define the mutation probability
        p = 0.1

        # Create a mask for which elements to mutate
        input_mask = np.random.rand(*input_genes.shape) < p
        output_mask = np.random.rand(*output_genes.shape) < p

        # Apply the mutation to the selected elements
        input_genes[input_mask] = self.mutate_np(input_genes[input_mask])
        output_genes[output_mask] = self.mutate_np(output_genes[output_mask])

        return input_genes, output_genes

    def can_eat(self, other):
        return self.energy >= 1.5 * other.energy and self.ancestor != other.ancestor


    def can_mate(self, other):
        return self.ancestor == other.ancestor and self.energy >= self.mature_energy and other.energy >= other.mature_energy #check baby time?

    def collision(self, other):
        dist = self.get_gap_distance(other)
        
        return dist < 0

    def eat(self, other):
        self.energy += other.energy
        other.energy = 0

    def can_sprint(self):
        return self.sprint_rest_timer >= sprint_rest_duration

    def step(self, agents, foods):
        baby = None
        self.update_state()
        self.update_speed_and_position()
        self.handle_lifespan()
        for agent in agents:
            if self == agent:
                continue
            distance = self.get_gap_distance(agent)
            if distance < 0:
                #check if we are eaten
                if agent.can_eat(self):
                    agent.eat(self)
                    return False, None, baby

                if self.can_mate(agent) and self.baby_rest_timer > BABY_TIME and self.energy > agent.energy and len(agents) < MAX_TOTAL_POPULATION:
                    baby = self.try_reproduce(agent)

                #agents should not pass through each other
                too_big = agent.body_size > self.body_size
                if too_big:
                    self.avoid(agent,distance)
        eaten_food_indices = self.eat_food(foods)
        
        still_alive = self.energy >= MIN_AGENT_ENERGY and self.age <= LIFESPAN
        return still_alive, eaten_food_indices, baby

    def update_state(self):
        self.energy *= ENERGY_DECAY_RATE
        self.body_size = self.calculate_body_size()
        self.speed = AGENT_SPEED_FACTOR / (1 + self.body_size)
        self.update_sprint()

    def calculate_body_size(self):
        return 0 if self.energy <= 0 else math.sqrt(self.energy) / ENERGY_TO_SIZE_FACTOR

    def update_speed_and_position(self):
        dx = np.cos(self.direction) * self.speed
        dy = np.sin(self.direction) * self.speed
        self.x = (self.x + dx) % ENV_SIZE
        self.y = (self.y + dy) % ENV_SIZE

    def update_sprint(self):
        if self.sprint:
            self.energy *= ENERGY_DECAY_RATE
            self.speed *= sprint_speed_multiplier
            self.update_sprint_duration()
        else:
            self.sprint_rest_timer = min(1+self.sprint_rest_timer, SPRINT_REST_DURATION)
            self.sprint_duration = max(self.sprint_duration - 0.5, 0)

    def update_sprint_duration(self):
        self.sprint_duration += 1
        if self.sprint_duration >= max_sprint_duration:
            self.end_sprint()

    def end_sprint(self):
        self.sprint = False
        self.sprint_duration = 0
        self.sprint_rest_timer = 0





    def get_gap_distance(self, other):
        if isinstance(other,Food):
            return food_distance_cache.get((self,other),999) - self.body_size - other.body_size
        else:
            return agent_distance_cache.get((self,other),999) - self.body_size - other.body_size



    def avoid(self, other, current_gap):
        jump_direction =  np.arctan2(self.y  - other.y, self.x - other.x)
        jump_distance = abs(current_gap)
        total_energy = self.energy + other.energy
        total_dx = np.cos(jump_direction) * jump_distance
        total_dy = np.sin(jump_direction) * jump_distance
        dx = total_dx * other.energy / total_energy
        dy = total_dy * other.energy / total_energy
        self.x = (self.x + dx) % ENV_SIZE
        self.y = (self.y + dy) % ENV_SIZE
        dx = -total_dx * self.energy / total_energy
        dy = -total_dy * self.energy / total_energy
        other.x = (other.x + dx) % ENV_SIZE
        other.y = (other.y + dy) % ENV_SIZE

    def eat_food(self, foods):
        eaten_food_indices = set()
        for i, food in enumerate(foods):
            if self.collision(food):
                self.mouth_open = False
                self.eat(food)
                eaten_food_indices.add(i)

        return eaten_food_indices


    def handle_lifespan(self):

        self.age += 1
        if self.energy > self.mature_energy:
            self.baby_rest_timer += 1



    def try_reproduce(self, agent):
        baby = None
        self.baby_rest_timer = 0
        self.energy -= self.baby_energy
        dx = np.cos(-self.direction) * self.body_size
        dy = np.sin(-self.direction) * self.body_size
        baby = Agent(self.x+dx, self.y+dy, self.direction, self.speed, self.baby_energy, self.mature_energy, self.ancestor,self.hue,agent, self,max(self.generation,agent.generation) + 1)
        self.children.append(baby)
        agent.children.append(baby)
        return baby

    def get_nearby_object_features(self, agents, foods):
        all_objects = agents + foods
        objects_in_vision = []
        all_features = []
        self.mouth_open = False
        for obj in all_objects:
            if self != obj:
                distance = self.get_gap_distance(obj)
                if distance < SIGHT_RADIUS:
                    all_features.append(self.get_relative_features(obj,distance))
                    objects_in_vision.append(obj)


        return objects_in_vision, np.array(all_features)

    def get_relative_features(self, other,distance_to_other):



        # Calculate direction from self to other
        dx, dy = other.x - self.x, other.y - self.y
        angle_to_other = np.arctan2(dy, dx) - self.direction
        # Ensure the angle is within [-pi, pi]
        angle_to_other = (angle_to_other + np.pi) % (2 * np.pi) - np.pi  

        if distance_to_other < self.body_size and self.can_eat(other):
            self.mouth_open = True
         
        distance_to_other = distance_to_other / SIGHT_RADIUS



        # Create vectors
        unit_vector_towards_other = np.array([dx, dy]) / distance_to_other
        other_speed_vector = np.array([other.speed * np.cos(other.direction), other.speed * np.sin(other.direction)])


        # Calculate speed towards or away from our agent
        speed_towards_us = np.dot(other_speed_vector, unit_vector_towards_other) / AGENT_SPEED_FACTOR

        # Get other's energy
        energy = other.energy / self.energy

        can_eat = 1.0 if other.can_eat(self) else 0.0
        can_be_eaten = 1.0 if self.can_eat(other) else 0.0
        same_species = 1.0 if self.ancestor == other.ancestor else 0.0
        can_mate = 1.0 if self.can_mate(other) else 0.0
        sprint_duration = self.sprint_duration / max_sprint_duration

        sprint_reset = self.sprint_rest_timer / SPRINT_REST_DURATION

        # Convert angles to cosine and sine components
        angle_to_other_cos = np.cos(angle_to_other)
        angle_to_other_sin = np.sin(angle_to_other)


        #we also need the direction the other is facing
        relative_facing_direction = other.direction - self.direction
        other_direction_cos = np.cos(other.direction)
        other_direction_sin = np.sin(other.direction)
        #return np.array([other_direction_cos,other_direction_sin, speed_towards_us,angle_to_other_cos, angle_to_other_sin, distance_to_other, energy, can_eat, can_be_eaten, same_species, sprint_duration, sprint_reset])
        return np.array([speed_towards_us,angle_to_other_cos, distance_to_other, energy, can_eat, can_be_eaten, can_mate, sprint_duration, sprint_reset])
    

    def angle_to_unit_vector(self, angle):
        x = np.cos(angle)
        y = np.sin(angle)
        return np.array([x, y])



    def execute_policy(self, action, action_object):
        #convert range to -1 to 1 representing away or towards
        action_toward_away = np.arctan(action[1])

        towards_direction = np.arctan2(action_object.y - self.y, action_object.x - self.x)
        away_direction = np.arctan2(self.y - action_object.y, self.x - action_object.x)

        #desired_direction = self.get_desired_direction(action_toward_away,towards_direction,action_object)
        desired_direction = towards_direction if action_toward_away > 0.1 else away_direction if action_toward_away < -0.1 else self.direction

        sprint = action[2] > 0
        #assign desired direction and sprint to agent
        self.direction = self.adjust_direction(desired_direction)
        self.sprint = self.can_sprint() and sprint





    def adjust_direction(self, desired_direction):
        delta_angle = (desired_direction - self.direction) % (2 * math.pi)
        if delta_angle > math.pi:
            delta_angle -= 2 * math.pi
        max_rotation = np.radians(max_turn_angle)
        delta_angle = np.clip(delta_angle, -max_rotation, max_rotation)
        return (self.direction + delta_angle) % (2 * math.pi)







class Custom2DEnvironment(gym.Env):
    def __init__(self, size):
        super(Custom2DEnvironment, self).__init__()
        self.size = size
        self.ancestor_id = 0
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=self.size, shape=(MIN_POPULATION, 4), dtype=np.float32)
        self.agents = []
        self.foods = []
        self.hue_names = {}
        self.reset()

    def random_initialize_agents(self, n, species_id):
        agent_hue = (species_id * 0.61803398875) % 1
        self.hue_names[species_id] = self.hue_to_name(agent_hue)
        agent_features = self.generate_random_agent_features(n, species_id,agent_hue)
        agents = [Agent(*features) for features in agent_features]
        return agents

    def generate_random_agent_features(self, n, species_id, hue):
        # You might need to adjust the distributions and bounds depending on your requirements
        # or use different distributions if required.
        agent_x = np.random.randint(0, self.size, size=n)
        agent_y = np.random.randint(0, self.size, size=n)
        agent_directions = np.random.uniform(-np.pi, np.pi, size=n)
        agent_speeds = np.random.uniform(0.01, 0.1, size=n)
        agent_baby_energies = np.abs(np.random.normal(0, 10, size=n)) + 15.0
        agent_mature_energies = agent_baby_energies * np.random.uniform(2, 5, size=n)
        agent_ancestors = np.full(n, species_id)
        agent_hues = np.full(n, hue)
        agent_features = list(zip(agent_x,agent_y, agent_directions, agent_speeds, agent_baby_energies, agent_mature_energies, agent_ancestors,agent_hues))

        
        return agent_features

    def reset(self):
        self.t = 0
        for i in range(NUM_SPECIES):
            self.agents.extend(self.random_initialize_agents(MIN_POPULATION, self.ancestor_id ))
            self.ancestor_id += 1
        self.foods = []
        return self.get_agent_observations()


    def update_distance_cache(self):
        # Collect agent coordinates into an array
        agent_coords = np.array([[agent.x, agent.y] for agent in self.agents])

        if len(self.foods) > 0:
            # Collect food coordinates into an array
            food_coords = np.array([[food.x, food.y] for food in self.foods])

            # Calculate the distances between all agents and all pieces of food in one go
            distances = cdist(agent_coords, food_coords)

            # Store the distances in the cache
            global food_distance_cache 
            food_distance_cache = {}
            for i, agent in enumerate(self.agents):
                for j, food in enumerate(self.foods):
                    food_distance_cache[(agent, food)] = distances[i, j]

        # Calculate and store distances between each pair of agents
        global agent_distance_cache
        agent_distance_cache = {}
        agent_distances = squareform(pdist(agent_coords))
        for i, agent1 in enumerate(self.agents):
            for j, agent2 in enumerate(self.agents):
                agent_distance_cache[(agent1, agent2)] = agent_distances[i][j]
                agent_distance_cache[(agent2, agent1)] = agent_distances[i][j]
        

    def step(self):
        self.t += 1
        self.update_distance_cache()

        self.get_policies()

        new_agents, all_eaten_food_indices = self.update_agents_state()
        self.agents = new_agents
        self.remove_eaten_foods(all_eaten_food_indices)
        rewards = self.calculate_rewards()

        self.spawn_agents_if_needed()
        self.spawn_food_if_needed()
        self.collect_and_display_stats()

        done = False
        return self.get_agent_observations(), rewards, done, {}




    def relu(self,x):
        return np.maximum(0,x)

    def get_policies(self):
        #input_weights = [agent.weight for agent in self.agents]
        #output_weights = [agent.weight for agent in self.agents]
        objects = []
        object_features = []
        agents_in_model = []
        for agent in self.agents:
            objects_in_vision, features = agent.get_nearby_object_features(self.agents, self.foods)
            if len(objects_in_vision) > 0:
                #agents_in_model.append(agent)
                #objects.append(objects_in_vision)
                #object_features.append(features)

                object_count = len(features)


                input_weights = agent.input_weights

                output_weights = agent.output_weights

                object_features = np.array(features)

                #add bias
                object_features = np.hstack((object_features,np.ones((object_count,1))))

                hidden_state = self.relu(np.matmul(object_features,input_weights))

                hidden_state = np.hstack((hidden_state,np.ones((object_count,1))))

                output = np.matmul(hidden_state,output_weights)
                
                #print('masked output ',masked_outputs)
                # Get the priority values for each agent and each object
                priority_values = output[:, 0]
                #print('priority ',priority_values)
                # Find the indices of the maximum priority value for each agent
                max_priority_index = np.argmax(priority_values)

                # Get the actions for the objects with the highest priority for each agent
                best_action = output[max_priority_index]
                best_action_objects = objects_in_vision[max_priority_index]

                agent.execute_policy(best_action, best_action_objects)




    def update_agents_state(self):
        new_agents = []
        all_eaten_food_indices = set()
        for agent in self.agents:
            still_alive, eaten_food_indices, baby = agent.step(self.agents, self.foods)
            if baby is not None:
                new_agents.append(baby)
            if still_alive:
                new_agents.append(agent)
                all_eaten_food_indices.update(eaten_food_indices)
            elif agent.energy < MIN_AGENT_ENERGY:
                turned_to_food = Food(agent.x, agent.y,agent.energy, image=agent.image, direction = agent.direction)
                self.foods.append(turned_to_food)
        return new_agents, all_eaten_food_indices

    def remove_eaten_foods(self, all_eaten_food_indices):
        self.foods = [food for i, food in enumerate(self.foods) if i not in all_eaten_food_indices]

    def calculate_rewards(self):
        rewards = np.zeros(MIN_POPULATION)
        # Add your reward calculation logic here if needed.
        return rewards

    def spawn_agents_if_needed(self):
        species_counts = Counter(agent.ancestor for agent in self.agents)
        total_count = np.sum(list(species_counts.values()))
        species_count = sum(1 for count in species_counts.values() if count > 1)

        #if species_count < 3 or total_count < 2 * MIN_POPULATION:
        #    self.agents.extend(self.random_initialize_agents(MIN_POPULATION, self.ancestor_id))
        #    self.ancestor_id += 1
        #    total_count += MIN_POPULATION
        for species_id, count in species_counts.items():
            if count < MIN_POPULATION and species_count <= NUM_SPECIES and count > 1:
                self.agents.extend(self.random_initialize_agents(MIN_POPULATION-count, species_id))
            if count > MAX_SPECIES_POP:
                hue = (self.ancestor_id * 0.61803398875) % 1
                self.hue_names[self.ancestor_id] = self.hue_to_name(hue)
                for agent in self.agents:
                    if agent.ancestor == species_id and random.random() < 0.5:
                        agent.ancestor = self.ancestor_id
                        agent.hue = hue
                        agent.set_image_tint()
                self.ancestor_id += 1


    def spawn_food_if_needed(self):
        if self.t % SECONDS == 0 and len(self.foods) < MAX_FOOD:
            for _ in range(FOOD_PER_SECOND):
                x, y = np.random.rand() * self.size, np.random.rand() * self.size
                energy = np.random.uniform(1, 10)
                new_food = Food(x, y, energy)
                self.foods.append(new_food)

    def collect_and_display_stats(self):
        if self.t % 10 * SECONDS == 0:
            stats_df = self.collect_stats()
            self.display_stats(stats_df)

    def collect_stats(self):
        # Collect stats from each agent
        stats_data = [{    'Ancestor' : agent.ancestor,
                    'Generation': agent.generation,
                    'Children': len(agent.children),
                    'BabyEnergy': round(agent.baby_energy),
                    'MatureEnergy': round(agent.mature_energy),
                    'Energy': round(agent.energy)} for agent in self.agents]
        stats_df = pd.DataFrame(stats_data)
        return stats_df

    def display_stats(self, stats_df):
        if self.t % (1 * MINUTES) == 0:
            self.plot_and_save_stats(stats_df,self.t)
            self.save_stats_to_csv(stats_df)

    def plot_and_save_stats(self, stats_df, t):
        if t % SAVE_INTERVAL == 0:
            sns.set_style("whitegrid")
            self.plot_energy_distribution(stats_df)
            self.plot_average_children(stats_df)
            self.plot_generation_distribution(stats_df)

    def plot_energy_distribution(self, stats_df):
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=stats_df, x='Ancestor', y='Energy')
        plt.title('Energy Distribution by Ancestor')
        plt.savefig('energy_distribution_by_ancestor.png')

    def plot_average_children(self, stats_df):
        plt.figure(figsize=(12, 6))
        sns.barplot(data=stats_df, x='Ancestor', y='Children', estimator=np.mean)
        plt.title('Average Number of Children by Ancestor')
        plt.savefig('Average Number of Children by Ancestor.png')

    def plot_generation_distribution(self, stats_df):
        generation_counts = stats_df.groupby(['Ancestor', 'Generation']).size().unstack().fillna(0)
        generation_counts.plot(kind='bar', stacked=True, figsize=(12, 6))
        plt.title('Generation Distribution by Ancestor')
        plt.xlabel('Ancestor')
        plt.ylabel('Number of Agents')
        plt.savefig('Number of Agents.png')


    def hue_to_name(self,hue):
        # Convert hue to RGB
        rgb = tuple(round(i * 255) for i in colorsys.hsv_to_rgb(hue, 1, 1))
        
        # Get closest CSS3 named color
        min_color_diff = None
        closest_color_name = None
        for name, hex_value in webcolors.CSS3_NAMES_TO_HEX.items():
            named_rgb = webcolors.hex_to_rgb(hex_value)
            color_diff = sum((a - b) ** 2 for a, b in zip(rgb, named_rgb))
            if min_color_diff is None or color_diff < min_color_diff:
                closest_color_name = name
                min_color_diff = color_diff

        return closest_color_name


    def get_largest_generations(self):
        return {ancestor: max(agent.generation for agent in self.agents if agent.ancestor == ancestor) for ancestor in set(agent.ancestor for agent in self.agents)}


    def save_stats_to_csv(self, stats_df):
        # Save the DataFrame to a CSV file
        stats_df.to_csv('stats.csv', index=False)

    def get_agent_observations(self):
        return np.array(
            [[agent.ancestor] for agent in self.agents],
            dtype=np.float32
        )

    def render(self, renderer):
        renderer.render(self)

    def close(self):
        if hasattr(self, 'renderer') and self.renderer is not None:
            self.renderer.close()
            self.renderer = None








class EnvironmentRenderer:
    def __init__(self, env, width, height):
        self.env = env
        self.width = width
        self.height = height
        self.pixels_per_unit = min(width, height) / ENV_SIZE
        self.view_center = np.array([ENV_SIZE / 2, ENV_SIZE / 2])

        self.zoom_level = 1
        self.drag_start_pos = None
        self.dragging = False
        self.mouse_down = False

        self.init_pygame()
        self.load_images()

    def init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

    def load_images(self):
        self.food_image = pygame.image.load('food.png')
        self.grass_image = pygame.image.load('grass.png')


    def world_to_screen(self, x, y):
        screen_x = ((x - self.view_center[0]) * self.pixels_per_unit) + (self.width / 2)
        screen_y = ((y - self.view_center[1]) * self.pixels_per_unit) + (self.height / 2)
        return int(screen_x), int(screen_y)

    def screen_to_world(self, screen_x, screen_y):
        world_x = ((screen_x - (self.width / 2)) / self.pixels_per_unit) + self.view_center[0]
        world_y = ((screen_y - (self.height / 2)) / self.pixels_per_unit) + self.view_center[1]
        return world_x, world_y

    def draw_background(self):
        grass_width_pixels, grass_height_pixels = self.grass_image.get_size()

        grass_scale = 500.0

        overlap = 1  # pixels
        #grass_width_pixels += overlap
        #grass_height_pixels += overlap
        # Calculate the grass size in world units
        grass_width_world = grass_width_pixels / grass_scale
        grass_height_world = grass_height_pixels / grass_scale

        # Calculate the scaled grass size in pixels
        scaled_grass_width_pixels = int(grass_width_pixels * self.pixels_per_unit / grass_scale + overlap)
        scaled_grass_height_pixels = int(grass_height_pixels * self.pixels_per_unit / grass_scale + overlap)
        
        # Scale the grass image
        scaled_grass_image = pygame.transform.scale(self.grass_image, (scaled_grass_width_pixels, scaled_grass_height_pixels))

        # Calculate the number of tiles needed to cover the screen
        tile_count_x = math.ceil(self.width / scaled_grass_width_pixels) + 1
        tile_count_y = math.ceil(self.height / scaled_grass_height_pixels) + 1

        # Calculate the starting tile coordinates in world units
        start_x_world = int((self.view_center[0] * self.pixels_per_unit - self.width / 2) // scaled_grass_width_pixels) * grass_width_world
        start_y_world = int((self.view_center[1] * self.pixels_per_unit - self.height / 2) // scaled_grass_height_pixels) * grass_height_world

        # Draw the grass image tiled across the screen
        for i in range(tile_count_x):
            for j in range(tile_count_y):
                world_x = start_x_world + i * grass_width_world
                world_y = start_y_world + j * grass_height_world
                screen_x, screen_y = self.world_to_screen(world_x, world_y)
                self.screen.blit(scaled_grass_image, (screen_x, screen_y))








    def draw_image(self,image,x,y,direction,body_size):
        scale_factor = body_size * self.pixels_per_unit / 120

        
        scaled_image = pygame.transform.scale(image, (int(image.get_width() * scale_factor), int(image.get_height() * scale_factor)))

        rotated_image = pygame.transform.rotate(scaled_image, -math.degrees(direction) + 90)

        screen_x, screen_y = self.world_to_screen(x, y)

        image_rect = rotated_image.get_rect(center=(screen_x, screen_y))
        self.screen.blit(rotated_image, image_rect)

    def inside_screen(self,other):
        world_left, world_top = self.screen_to_world(0, 0)
        world_right, world_bottom = self.screen_to_world(self.width, self.height)
        buffer = other.body_size
        return world_left - buffer <= other.x <= world_right + buffer and world_top - buffer <= other.y <= world_bottom + buffer
                

    def draw_agents(self):
        for agent in self.env.agents:
            if self.inside_screen(agent):



                self.draw_image(agent.image,agent.x,agent.y,agent.direction,agent.body_size)
                image = FACE_MOUTH_OPEN_IMAGE if agent.mouth_open else FACE_MOUTH_CLOSED_IMAGE
                self.draw_image(image,agent.x,agent.y,agent.direction,agent.body_size)
                
                #screen_x, screen_y = self.world_to_screen(agent.x, agent.y)

                #draw red circle around body
                #pygame.draw.circle(self.screen, (255, 0, 0), (screen_x, screen_y), int(agent.body_size * self.pixels_per_unit), 1)
                
                #draw blue circle around sight radius
                #sight_radius_pixels = int((SIGHT_RADIUS + agent.body_size)* self.pixels_per_unit)
                #pygame.draw.circle(self.screen, (0, 0, 255), (screen_x, screen_y), sight_radius_pixels , 1)


    def draw_food(self):
        for food in self.env.foods:
            if self.inside_screen(food):
                image = food.image
                self.draw_image(image,food.x,food.y,food.direction,food.body_size)            
                screen_x, screen_y = self.world_to_screen(food.x, food.y)
                #radius = int(food.body_size * self.pixels_per_unit * self.zoom_level)
                #pygame.draw.circle(self.screen, (0, 255, 0), (screen_x, screen_y), radius,1)


    def handle_event(self, event):
        if event.type == pygame.QUIT:
            pygame.quit()
            return
        elif event.type == pygame.MOUSEWHEEL:
            self.handle_zoom(event)
        elif event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
            self.handle_pan(event)

    def handle_key_press(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            self.view_center[1] -= pan_speed / self.pixels_per_unit
        if keys[pygame.K_s]:
            self.view_center[1] += pan_speed / self.pixels_per_unit
        if keys[pygame.K_a]:
            self.view_center[0] -= pan_speed / self.pixels_per_unit
        if keys[pygame.K_d]:
            self.view_center[0] += pan_speed / self.pixels_per_unit

    def render(self, fps=SECONDS):
        for event in pygame.event.get():
            self.handle_event(event)

        self.handle_key_press()
        self.update_view_center()
        self.draw_background()  
        self.draw_agents()
        self.draw_food()
        
        pygame.display.flip()
        self.clock.tick(fps)

    def update_view_center(self):
        self.view_center[0] = self.view_center[0] % self.env.size  # Wrap x-coordinate
        self.view_center[1] = self.view_center[1] % self.env.size  # Wrap y-coordinate

    def handle_zoom(self, event):
        if event.type == pygame.MOUSEWHEEL:
            if event.y > 0:  # Scrolling up
                self.pixels_per_unit *= 1.25
            elif event.y < 0:  # Scrolling down
                self.pixels_per_unit /= 1.25

    def handle_pan(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.drag_start_pos = np.array(event.pos)
        elif event.type == pygame.MOUSEMOTION and self.drag_start_pos is not None:
            delta = (np.array(event.pos) - self.drag_start_pos) / self.pixels_per_unit
            self.view_center -= delta
            self.drag_start_pos = np.array(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.drag_start_pos = None



    def close(self):
        pygame.quit()


def main():
    env = Custom2DEnvironment(size=ENV_SIZE)
    renderer = EnvironmentRenderer(env, 1920, 1000)
    t=0
    try:
        running = True
        while running:
            t += 1
            env.step()  # Pass dummy actions, since we don't use them for now
            renderer.render()
            if t % 90 == 0:
                print("agents: ",len(env.agents),"foods: ",len(env.foods)," FPS:", renderer.clock.get_fps())
                species_counts = Counter(agent.ancestor for agent in env.agents)
                larget_generations = env.get_largest_generations()
                for key, val in list(species_counts.items()):
                    color_name = env.hue_names[key]
                    gen = larget_generations[key]
                    print("speciesID: ",key,"\tpop: ",val,"\tmax gen: ",gen,"\tcolor: ",color_name)

            #if t > 1 * MINUTES:
            #    running = False
            #    break

            # Handle user input to exit gracefully
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
    finally:
        renderer.close()

import cProfile
import pstats


#profiler = cProfile.Profile()
#profiler.enable()

main()

#profiler.disable()
#stats = pstats.Stats(profiler).sort_stats('tottime')  # You can sort by other parameters like 'cumtime'
#stats.print_stats(10)  # Prints only the top 10 lines