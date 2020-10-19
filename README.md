# Heuristics

## Week-1 : Expanding Nim Game
Game can be found [here](https://cs.nyu.edu/courses/fall20/CSCI-GA.2965-001/expandingnim.html) in the course web site of Heursitics class taught by Prof. Shasha. Here is a description of the game copied from the course web site.
```
You and your opponent are presented with some number of stones that I will announce the day of the competition. The winner removes the last stone(s). The first player can take up to three (1, 2, or 3). At any later point in play, call the maximum that any previous player has taken currentmax . Initially currentmax has the value 0. At any later turn, a player may take (i) 1, 2, or 3 if a reset has been imposed by the other player in the immediately preceding turn (ii) up to a maximum of 3 and 1 + currentmax, otherwise. Thus, a reset by player P affects P's opponent's following move, but does not change currentmax nor does it affect any move after that.

To see how this changes the strategy from normal nim, suppose there are 8 stones to begin with. In normal Nim, the second player can force a win. Suppose the first player removes 1, 2, or 3 stones. The second player removes 3, 2, or 1 respectively, leaving the first player with four stones. If the first player removes 1, 2, or 3 stones at this point, the second player can remove the rest. However, in expanding nim, if the first player removes one stone and the second player removes three, the first player can win by removing all four that remain.

Here is another example just to show you the mechanics: if the first player removes 3, the second player can remove up to 4, in which case the first player can remove any number of stones up to and including 5.
In our tornament, two teams will play expanding nim with a reset option against each other. I will provide the initial number of stones (under 1,000). Each team will play once as the first player and once as the second player. The team may use the reset option at most four times in each game. The reset option permits a team after making its move to force the maximum number of stones that can be removed in the next turn for the other team (and in the next turn only) to be three again. After that turn play continues using the currentmax until and if some team exercises its reset option.

Hint: dynamic programming is a good idea, but you must keep track of which player's turn it is, how many stones are left, and what the currentmax is, and who has used the reset option and how often.

```

I started with a case no reset and no expansion, which we can look into the future. The nature of game allows to force the opponent to a loosing state.

For the expansion I used a dynamic programming solution looking into two steps depth and trying to maximize the outcome after two states. 

## Week-2 : Optimal Touring
The problem can be found [here](https://cs.nyu.edu/courses/fall20/CSCI-GA.2965-001/tour.html) in the course web site of Heursitics class taught by Prof. Shasha. Here is a description of the game copied from the course web site.
```
You want to visit up to n sites over k days. Each site has certain visiting hours. You have fixed a time you want to spend at each site which must all happen in one day. The time to go from site to site in minutes is the sum of street and avenue differences between them. On each day, you can start at any site you like (as if you teleported from the previous place you visited and slept on the street).

No more than 10 days and 200 sites.

You will be told the statistics of the sites and the number of days on the day of the contest.

The format will be
site, x location, y location, desiredtime, value
site, day, beginhour, endhour

Here is a typical file.

You want to achieve the maximum possible value summed over all sites within the time constraints. Visiting a site twice or more times gives no more value than visiting it once.

Note that a pure greedy strategy wouldn't be so good. Such a strategy might have you visit a site and and then visit the next nearest site. That might lead to large number of sites being found the first day (e.g. a central circle), but then later days might not have sites close to one another.

Your output should simply give a sequence of sites in the order of visit per day, one line per day.

```

First step, I split the sites to number of days groups, I was lazy and didn't want to come up with a new algorithm to split optimally or at least very good, I just use k-means clustering.

Second step, I use hungarian matching to match the days and sites according to a simple metric which is the sum of total service hours of each site for each day. 

Third step, I use 2-opt traveling salesman algorithm to find the best circle to trace all the nodes.

Fourth step, I iterate over this circle and use each node as my starting point, then from this node onwards note all the sites I can visit on the circle. I do this also in reverse order since the reverse order is also a solution of TSP.

Fifth step, I take the best route for each day. 

I repeat steps 1-5 as much as time permits, sometimes since the k-means is random the algorithm can come up with a better solution. So there is a improvement there to make as a future work.


## Week-3 : Balancing the Board (No Tipping Game)

The problem can be found [here](https://cs.nyu.edu/courses/fall20/CSCI-GA.2965-001/notipping.html) in the course web site of Heursitics class taught by Prof. Shasha. Here is a description of the game copied from the course web site.
```
Given a uniform, flat board (made of a titanium alloy) 60 meters long and weighing 3 kilograms, consider it ranging from -30 meters to 30 meters. So the center of gravity is at 0. We place two supports of equal heights at positions -3 and -1 and a 3 kilogram block at position -4.

The No Tipping game is a two person game that works as follows: the two players each start with k blocks having weights 1 kg through k kg where 2k is less than 50. The first player places one block anywhere on the board, then the second player places one block anywhere on the board, and play alternates with each player placing one block until the second player places his or her last block. (The only allowable positions are on meter markers. No player may place one block above another one, so each position will have at most one block.) If after any ply, the placement of a block causes the board to tip, then the player who did that ply loses. Suppose that the board hasn't tipped by the time the last block is placed. Then the players remove one block at a time in turns. At each ply, each player may remove a block placed by any player or the initial block. If the board tips following a removal, then the player who removed the last block loses.

As the game proceeds, the net torque around each support is calculated and displayed. The blocks, whether on the board or in the possession of the players, are displayed with their weight values. The torque is computed by weight times the distance to each support. Clockwise is negative torque and counterclockwise is positive torque. You want the net torque on the left support to be non-positive and the net torque on the right support to be non-negative. Here was a strategy that worked well in 2016.

```

First step, building up the board, in other words, placing the weights on the board without tipping. Here I used a simple strategy, for the heaviest weight I have in my inventory, I pick the rightmost spot which will not cause tipping. This way I try to force the center of mass to the rightmost place which is still left of the right support. 

Second step, is to use a tree search (minmax) but with a shallow depth. Otherwise it will take too long to run. Here it ry to have the board cm as far right as possible in order to survive until the rounds that i can actually compute the entire playout.

Third step, i am using the entire game future states to have a playing strategy, if i can force a win with any of my actions i use that action otherwise try to move a state that can force a win in the future.





## Week-4 : Ambulance Pick up

The problem can be found [here](https://cs.nyu.edu/courses/fall20/CSCI-GA.2965-001/ambulance.html) in the course web site of Heursitics class taught by Prof. Shasha. Here is a description of the game copied from the course web site.
```
The ambulance planning real-time problem is to rescue as many people as possible following a disaster. The problem statement identifies the locations of people and the time they have to live. You can also establish mobile hospitals at the beginning of the problem. The problem is to get as many people to the hospitals on time as possible.

In our case, the graph is the Manhattan grid with every street going both ways. It takes a minute to go one block either north-south or east-west. Each hospital has an (x,y) location that you can determine when you see the distribution of victims. The ambulances need not return to the hospital where they begin. Each ambulance can carry up to four people. It takes one minute to load a person and one minute to unload up to four people. Each person will have a rescue time which is the number of minutes from now when the person should be unloaded in the hospital to survive. By the way, this problem is very similar to the vehicle routing problem about which there is an enormous literature and nice code like "jsprit" which was used in 2015 to great effect. If anyone wants to take a break from programming, he/she may volunteer to look up that literature and propose some good heuristics.

So the data will be in the form:
person(xloc, yloc, rescuetime)

```

First step, clustering the patients and deploying hospitals to the centers without assigning num ambulances yet.

Second step, building up the possible up to quadruple of patients, which are feasible to rescue at one turn. (H-P1-H, H-P1-P2,H ... )

Third step, we have max ambulance set, which is the largest number of ambulance in the given scenario for any hospital. Using this number, create routes for every hospital as if each hospital has this max number of ambulances.
  * Two methods, one forward and one backward.

Fourth step, match the ambulance numbers such that remove some ambulance routes from every hospital. Find the maximum matching.






## Week-5 : Random Lawn Mower
The problem can be found [here](https://cs.nyu.edu/courses/fall20/CSCI-GA.2965-001/randomower.html) in the course web site of Heursitics class taught by Prof. Shasha. Here is a description of the game copied from the course web site.
```
There is one perfectly flexible but unbreakable rope attached to two posts on a large but unkepmt lawn. The posts are a distance 1000 meters apart and the rope is of length 1100 meters. So there is some slack in the rope.

A randomower is a driverless lawnmower that can change directions at any time by an arbitrary angle. Fortunately, it can be clipped to the rope so it won't wander off too far. You are trying to use this random lawnmower to cut at least part of the grass between the posts. We'll call the rope's zero point is at post A and its end point at post B is r. The distance between A and B is d where d < r.

To train your intuition, suppose you wanted to cut the grass to clear a path, i.e. a line segment the width of the lawnmower itself, between the two posts. It's ok if more grass is cut on the sides in addition, but you want to be sure to have a continuous path. The challenge is to do this with the minimum number of attachments.

Warm-up: What is the smallest number of attachments necessary to mow a straight line segment (and, optionally, other grass) from post to post?

Solution: Call the difference r - d, diff. Attach the randomower to the diff meter mark along the rope starting say from post A. Because the rope is diff meters longer than the distance between the posts, the distance along the rope between the attachment point of the randomower and post B is d. Therefore, the randomower can reach post A up to a point diff from point A in the direction of B. Next attach the randomower at the 2*diff meter mark along the rope from post A. So, all together we will need ceiling(d/diff) attachments.

Because of its random movements, the randomower will mow more grass than what is in the line segment. That suggests a game.

Suppose that T1 and T2 play a game in which each wants to use attachments that cuts as much of the lawn as possible. They take turns as follows: T2 makes the first attachment. Then T1 makes two. Then T2 makes two. This goes on until each makes the same number of moves (T2, in the last move, makes one attachment). Each player gets credit for every part of the lawn that is first mowed by randomower due to an attachment by that player.

```

For each time when pick a pair of attachment, I search entire space with 10 step-size, so 110^3 possibility. I check the values as first + second - third attachment. As the next player will be doing the third attachment. I pick max-min to get the best move among what I explored. 
