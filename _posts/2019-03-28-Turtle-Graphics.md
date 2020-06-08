# Logo programming language and Turtle Graphics

I started programming with the Logo programming language in high school and it was very fun using its various features specially turtle graphics to create interesting geometric patterns. It is one of the most intuitive way of learning to program since the language offers simple syntax while providing enough framework to make complex programs as well, also the graphics are inbuilt in the language.

The program's output is displayed on the screen when the program is executed so you would instantly know if the program is wrong because it would create a different shape than what you programmed it for.

Logo is an educational programming language and the name is derived from the Greek logos meaning word or thought. Logo is a general-purpose language but it is widely known for its use of turtle graphics.

In this post I will describe about one of the most interesting and useful feature of the Logo, the turtle graphics and its modern day equivalent available in Python.

## About Turtle
The **turtle** acts as a cursor on the screen which reacts to commands and will mark the path on which it transits by the pen.
The turtle can be described by the following properties
- Location on the plane
- Orientation of the turtle 
- A Pen which can be On or Off 

The Python programming language's standard library includes a Turtle graphics module we will use it to implement various geometric patterns.

We can import the turtle graphics library by

{% highlight python %}

import turtle

{% endhighlight %}

And we can create a turtle instance by any name such as leo as follows
{% highlight python %}

leo = turtle.Turtle()

{% endhighlight %}

The following skeleton/boilerplate code will be used in the examples below

{% highlight python %}

import turtle
wn = turtle.Screen()
leo = turtle.Turtle()

# drawing commands can be placed here. For example : leo.forward(50)

wn.mainloop()

{% endhighlight %}

## Step One: Starting with the Basics
### The Line
Our turtle named leo start at (0, 0) in $$xy$$ plane. We can draw a line by using the command **leo.forward(50)** which will move our turtle leo 50 pixel in the direction where leo is pointing that is positive x direction by default. 

{% highlight python %}

leo.forward(50)

{% endhighlight %}

[![Line](/assets/turtle_graphics/line.png)](/assets/turtle_graphics/line.png)

### The Square
Before drawing a square we have to understand the relative position and orientation of the turtle.

When we move turtle forward as in case of previous example the turtle is now facing the positive axis, so we have to move the turtle in upward direction to make a perpendicular to the first line.

We will acheive this by rotating the turtle in counterclockwise direction by 90 degrees by  **leo.left(90)**. Now the turtle is facing upward direction finally move the turtle forward.

To draw a square we just have to repeat this four times.
We define the following function which takes turtle object and side length as arguments

{% highlight python %}

def draw_square(turtle, side_len=50):
    for i in range(4):
        turtle.forward(side_len)
        turtle.left(90)    
        
{% endhighlight %}


[![Square](/assets/turtle_graphics/sq.png)](/assets/turtle_graphics/sq.png)

### A Regular polygon

To generalize the earlier square method to generate any regular polygon we will slightly modify our square function. Remember the interior angle of the n sided regular polygon is $$ ((n - 2) * 180) / n $$ and relative to the turtle position it can be simplified by using the relative orientation of turtle $$ 180 - ((n - 2) * 180) / n $$ or simply $$ 360 / n $$ 

{% highlight python %}

def draw_reg_poly(turtle, sides=3, side_len=50):
    interior_angle = 360 / sides
    for i in range(sides):
        turtle.forward(side_len)
        turtle.left(interior_angle)
    
{% endhighlight %}

Testing with various values of n we get for
- n := 3

[![Triangle](/assets/turtle_graphics/triangle.png)](/assets/turtle_graphics/triangle.png)

- n := 5

[![Pentagon](/assets/turtle_graphics/pentagon.png)](/assets/turtle_graphics/pentagon.png)

- n := 6

[![Hexagon](/assets/turtle_graphics/hex.png)](/assets/turtle_graphics/hex.png)


### The Star
To create a star pattern by hand we first draw a straight horizontal line then a line at an acute angle and repeating until the star is formed. Now if we find the angle we can draw the star. Can you find the angle?  
> Hint: The central shape formed in star is a pentagon.

Let's implement the star pattern

{% highlight python %}

def star(turtle, side_len=50):
    for i in range(5):
        turtle.forward(side_len)
        turtle.left(144)

{% endhighlight %}

[![Star](/assets/turtle_graphics/star.png)](/assets/turtle_graphics/star.png)


## Step Two: Creating cool patterns
Now that we know the basics we can create interesting patterns by augmenting the methods we have already made. 

### Create a pattern by rotating the polygon
We already have a function **draw_reg_poly()** now we just have to rotate our turtle before calling the function.

Let's rotate pentagon.

{% highlight python %}

for i in range(20):
    leo.left(18)
    draw_reg_poly(5)

{% endhighlight %}

[![Rotated_Polygon](/assets/turtle_graphics/rotate_poly.png)](/assets/turtle_graphics/rotate_poly.png)


Now we have a complex structure just by rotating the polygon, it can be any shape square, triangle, hexagon etc.

### Spirals
To create spiral we are going to decrease the size of the side length at each iteration by a fixed amount so that the figure wraps neatly into a spiral

{% highlight python %}
def spiral(turtle):
    side = 400
    for i in range(100):
        turtle.forward(side)
        turtle.left(90)
        side = side - 4
{% endhighlight %}

[![square_spiral](/assets/turtle_graphics/square_spiral.png)](/assets/turtle_graphics/square_spiral.png)


We can add colors to the spiral by defining a list of colors since we are drawing a square spiral we will need four colors.
The choice of color can be specified by general color name such as blue, yellow or specific colorstring.
{% highlight python %}
def spiral(turtle):
    colors = ['green', 'blue', 'red', 'yellow']
    for i in range(100):
        turtle.pencolor(colors[i % 2])
        turtle.forward(side)
        turtle.left(90)
        side = side - 4
{% endhighlight %}

[![square_spiral_color](/assets/turtle_graphics/square_spiral_col.png)](/assets/turtle_graphics/square_spiral_col.png)


Another variant of spiral can be created by varying the angle instead of using perpendiculars.

{% highlight python %}
def spiral(turtle):
    for i in range(100):
        turtle.forward(side)
        turtle.left(92)
        side = side - 4

{% endhighlight %}

[![sq_spiral](/assets/turtle_graphics/sq_spiral.png)](/assets/turtle_graphics/sq_spiral.png)

Similarly we can create triangular spiral, by changing the angle of the previous program to 121 degrees. Try different values and see what happens.

[![tr_spiral](/assets/turtle_graphics/tr_spiral.png)](/assets/turtle_graphics/tr_spiral.png)

There are so much more designs to explore as we have just scratched the surface of the turtle graphics library. Multitude of patterns can be implemented by turtle graphics such as Lindenmayer system, Diagrams, Trees, Fractals, Animations etc.
So try to experiment and create your own designs.


Refrences and further readings:
1. [https://docs.python.org/3/library/turtle.html](https://docs.python.org/3/library/turtle.html)
2. [https://en.wikipedia.org/wiki/Turtle_graphics](https://en.wikipedia.org/wiki/Turtle_graphics)
3. [https://python.camden.rutgers.edu](https://python.camden.rutgers.edu/python_resources/python3_book/hello_little_turtles.html)
