#include <GL/gl.h>
#include <GL/glut.h>
//#include <GL/gl.h>
//#include "GL/glext.h"

//#include <stdlib.h>
#include <iostream>

//#include <string>
//#include <fstream>

//#include <experimental/random>
#include <time.h>

unsigned char *img;
int width=320, height=320;

int t=0;

void display() {
   glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);

   for (int i=0; i<width*height; i++) {
     img[i*4 + 0] = 50+t;
     img[i*4 + 1] = 50+t;
     img[i*4 + 2] = 50+t;
     img[i*4 + 3] = 255;
   }
   t++;
   std::cout << t << std::endl;
   glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, img);
 
   glFlush(); // Render now
}

void idle_func( void ) {
  int t1 = (int)time(NULL);
  int t2 = (int)time(NULL);
  while (t2 - t1 < 1000) {
    std::cout << t2 - t1 << std::endl;
  }

  glutPostRedisplay();
}
 
int main(int argc, char** argv) {

  //img = (unsigned char*) malloc((size_t)(width*height*3)*sizeof(int));
  img = new unsigned char[width * height * 4];


  for (int i=0; i<width*height; i++) {
    img[i*4 + 0] = 50;
    img[i*4 + 1] = 50;
    img[i*4 + 2] = 50;
    img[i*4 + 3] = 255;
  }

  glutInit(&argc, argv);
  glutCreateWindow("Window");
  glutInitWindowSize(width, height);
  glutDisplayFunc(display);
  glutIdleFunc(idle_func);
  glutMainLoop();
  return 0;
}
