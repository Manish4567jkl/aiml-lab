% Facts
parent(john, mary).
parent(mary, alice).

% Rules
sibling(X, Y) :- parent(Z, X), parent(Z, Y), X \= Y.