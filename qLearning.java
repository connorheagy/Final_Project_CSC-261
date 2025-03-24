
    
    public class qLearning {
        private static final int SIZE = 16;
        private static final int NUMSIMULATIONS = 10000;
        private static final int NUMEPISODES = 1000;
        private double alpha;    // Learning rate
        private double gamma;    // Discount factor
        private double epsilon;  // Exploration rate
        private int numSimulations; //number of simulations
        private int numEpisodes; // number of episodes
        double[][] simulationResults = new double[NUMSIMULATIONS][NUMEPISODES];
        
        private final double[][] qTable;  // Q-table: state × action
        private final int[] actions = {1, -1, 4, -4};  // right, left, down, up
        
    
        public qLearning(double alpha, double gamma, double epsilon, int simulations, int episodes) {
            this.alpha = alpha;
            this.gamma = gamma;
            this.epsilon = epsilon;
            this.numSimulations = simulations;
            this.numEpisodes = episodes;
            this.qTable = new double[SIZE][actions.length];
            //innitialize simulation results
            for(int i = 0; i < numSimulations; i++){
                for(int j = 0; j < numEpisodes; j++){
                    simulationResults[i][j] = 0;
                }
            }
            initializeQTable();
        }
        
        private void initializeQTable() {
            for (int s = 0; s < SIZE; s++) {//16
                for (int a = 0; a < actions.length; a++) {//by 4
                    qTable[s][a] = Math.random() * 0.1;
                }
            }
        }
        
        private int chooseAction(int state, double currentEpsilon) {
            // Epsilon-greedy action selection with current epsilon value
            if (Math.random() < currentEpsilon) {
                return (int)(Math.random() * actions.length);
            } else {
                return getBestAction(state);
            }
        }
        
        private int getBestAction(int state) {
            int bestAction = 0;
            double bestValue = qTable[state][0];
            
            for (int a = 1; a < actions.length; a++) {
                if (qTable[state][a] > bestValue) {
                    bestValue = qTable[state][a];
                    bestAction = a;
                }
            }
            return bestAction;
        }
        
        private boolean isValid(int current, int next) {
            if (next < 0 || next >= SIZE) return false;
            if (current % 4 == 0 && next == current - 1) return false;
            if ((current + 1) % 4 == 0 && next == current + 1) return false;
            return true;
        }
        
        private boolean isTerminal(int state) {
            return state == 0 || state == SIZE - 1;
        }
    
    
    
        /**
         * Train the Q-learning agent with configurable parameters
         * @param alpha Learning rate (0 to 1)
         * @param gamma Discount factor (0 to 1)
         * @param epsilon Initial exploration rate (0 to 1)
         * @param numSimulations Number of complete training simulations to run
         * @param numEpisodes Number of episodes per simulation
         * @param decayEpsilon Whether to decay epsilon over episodes
         * @return Average rewards per episode across all simulations
         */
        public double[][] train( boolean decayEpsilon, boolean decayAlpha) {
            
            //create a table for simulation results
            
            
            //for each simulation..
            for (int sim = 0; sim < numSimulations; sim++) {
                // Reset Q-table for each simulation
                initializeQTable();
                
                //for each episode.., find average episode taken per simulation
                /* supposed to count number of steps it took,  */
                for (int episode = 0; episode < numEpisodes; episode++) {
    
    
                    // Calculate decaying epsilon if enabled
                    //this will allow the epsilon to change based on the number of episodes.
                    double currentEpsilon = decayEpsilon ? 
                        epsilon * (numEpisodes - episode) / numEpisodes : 
                        epsilon;//decaying epsilon if enabled (for the)
                    
                    // Calculate decaying alpha if enabled
                    //this wil allow the alpha to change based on number of episodes
                    double currentAlpha = decayAlpha ? 
                        alpha * (numEpisodes - episode) / numEpisodes : 
                        alpha;
                
                    // Track episode statistics
                    double episodeReward = 0;
                    int steps = 0;
                    
                    // Start from state 12
                    int state = (int)(12);
    
                    //if we want to make the states random.
                    // while (isTerminal(state)) {
                    //     state = (int)(Math.random() * SIZE);
                    // }
                    
                    //while the state isn't terminal and the steps are less than the 32, we can take that many steps.
                    while (!isTerminal(state) && steps < SIZE * 2) {  // Limit steps to prevent infinite loops
                        //iterate the steps each loop over
                        steps++;
                        
                        // Choose and take action
                        int actionIndex = chooseAction(state, currentEpsilon);
                        int action = actions[actionIndex]; //using action index to select action
                        int nextState = state + action; // next action chosen through simple addition of the action-index
                        
                        // Calculate reward
                        double reward = -1.0;  // Default step cost, this can change if the reward is 0.
                        if (!isValid(state, nextState)) {
                            nextState = state;  // Invalid move
                            reward = -2.0;      // Penalty for invalid move
                        } else if (isTerminal(nextState)) {
                            reward = 0.0;       // Reward for reaching goal
                        }
                        
                        // Update total reward per episode
                        /*maybe this is where we can sum for a line graph? */
                        episodeReward += reward;
    
                        
                        // Q-learning update
                        double currentQ = qTable[state][actionIndex]; // show the current q-table
    
                        //initialize the next Q, the MAX next Q 
                        double maxNextQ = 0.0; 
                        //next value has to be terminal
                        if (!isTerminal(nextState)) {
                            //select bext action based on next state, and best action based on the get best action function 
                            maxNextQ = qTable[nextState][getBestAction(nextState)];
                        }
                        
                        //the qtable is updated with the central formula
                        qTable[state][actionIndex] = currentQ + 
                            currentAlpha * (reward + gamma * maxNextQ - currentQ);
                        
                        //set the next state
                        state = nextState;
                    }
                    
                    // Store episode results
                    simulationResults[sim][episode] = episodeReward;
                }
            
                if ( sim % 10000 == 0 ){
                    // System.out.printf("Sim Count: %d", sim);
                    // this.printPolicy();
                }
            }//for simulatinos
    
            
            return simulationResults;
        }
        
        /* sums average reinforcement per simulation */
        public double averageReinforcement(){
            double average;
            double sum = 0;
    
            for(int i = 0; i < NUMSIMULATIONS; i++){
                for(int j = 0; j < NUMEPISODES; j++){
                    sum += simulationResults[i][j];
                }
            }
    
            average = (sum / (NUMSIMULATIONS * NUMEPISODES));
            return average;
        }
    
        // /* sums average reinforcement per simulation */
        // public double averageReinforcementUntilN(int n){
        //     double average;
        //     double sum = 0;
    
        //     for(int i = 0; i < NUMSIMULATIONS; i++){
        //         for(int j = 0; j < n; j++){
        //             sum += simulationResults[i][j];
        //         }
        //     }
    
        //     average = (sum / (NUMSIMULATIONS * NUMEPISODES));
        //     return average;
        // }
    
        // public void averageROverTime(){
            
        //     for(int i = 0; i < NUMEPISODES; i++){
        //         System.out.print(averageReinforcementUntilN(i));
        //     }
        // }
    
        public void printPolicy() {
            System.out.println("Learned Policy:");
            for (int i = 0; i < SIZE; i++) {
                if (isTerminal(i)) {
                    System.out.print("* ");
                } else {
                    int bestAction = actions[getBestAction(i)];
                    String arrow = switch (bestAction) {
                        case 1 -> "→ ";
                        case -1 -> "← ";
                        case 4 -> "↓ ";
                        case -4 -> "↑ ";
                        default -> "? ";
                    };
                    System.out.print(arrow);
                }
                if ((i + 1) % 4 == 0) System.out.println();
            }
        }
        
        public static void main(String[] args) {
            
            
            // // Print average rewards across all simulations
            // System.out.println("Training Results:");
            // for (int episode = 0; episode < results[0].length; episode += 100) {
            //     double avgReward = 0;
            //     for (int sim = 0; sim < results.length; sim++) {
            //         avgReward += results[sim][episode];
            //     }
            //     avgReward /= results.length;
            //     System.out.printf("Episode %d: Average Reward = %.2f%n", 
            //                     episode, avgReward);
            // }
    
            // Example usage with different parameters
            System.out.println("Exp 1:");
            qLearning exp1 = new qLearning(0.1, 0.9, 0.25, NUMSIMULATIONS, NUMEPISODES);
            exp1.train(false, false);
            exp1.printPolicy();
            System.out.println("average reinforcement experiment 1: " + exp1.averageReinforcement());
    
    
            System.out.println("Exp 2:");
            qLearning exp2 = new qLearning(1, 0.9, 1, NUMSIMULATIONS, NUMEPISODES);
            exp2.train(true, true);
            exp2.printPolicy();
            System.out.println("average reinforcement experiment 2: " + exp2.averageReinforcement());
    
            System.out.println("Exp 3:");
            qLearning exp3 = new qLearning(0.1, 0.9, 1, NUMSIMULATIONS, NUMEPISODES);
            exp3.train(true, false);
            exp3.printPolicy();
            System.out.println("average reinforcement experiment 3: " + exp3.averageReinforcement());
    
            System.out.println("Exp 4:");
            qLearning exp4 = new qLearning(1, 0.9, 0.1, NUMSIMULATIONS, NUMEPISODES);
            exp4.train(false, true);
            exp4.printPolicy();
            System.out.println("average reinforcement experiment 4: " + exp4.averageReinforcement());
    
            System.out.println("Exp 5:");
            qLearning exp5 = new qLearning(0.1, 0.9, 0.1, NUMSIMULATIONS, NUMEPISODES);
            exp5.train(false, false);
            exp5.printPolicy();
            System.out.println("average reinforcement experiment 5: " + exp5.averageReinforcement());
        }
    }
