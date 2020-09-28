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



