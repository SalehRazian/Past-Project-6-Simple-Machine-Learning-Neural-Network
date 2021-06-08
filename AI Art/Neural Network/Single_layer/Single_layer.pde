int[][] input = {{0,0,1},{1,0,1},{1,0,1},{0,1,1}};
int[] object = {0,1,0,1};
float[] weight = new float[3];

void setup(){
  size(700,700);
  for(int i=0;i<3;i++){
    weight[i]=(float)Math.random();
  }
}

float[] output = {0,0,0,0};
float[] adj = {0,0,0,0};

void draw(){
  clear();
  background(50);
  for(int i=0;i<4;i++){
    for (int j=0;j<3;j++){
      float x = input[i][j] * weight[j];
      output[i]+= x;
      noStroke();
      fill(255*input[i][j]);
      ellipse(100,50+(100*i)+(20*j),10,10);
      fill(255*weight[j]);
      ellipse(200,100+(100*j),10,10);
      stroke(255*x);
      line(100,50+(100*i)+(20*j),200,100+(100*j));
      line(200,100+(100*j),300,70+(100*i));
    }
    output[i] = (float)(1/(1+Math.exp(-(output[i]))));
    noStroke();
    fill(255*output[i]);
    ellipse(300,70+(100*i),10,10);
  }
  for(int i=0;i<4;i++){
    float x = object[i] - output[i];
    float y = output[i] * (1 - output[i]);
    adj[i] = x * y;
    noStroke();
    fill(255*object[i]);
    ellipse(400,70+(100*i),10,10);
    fill(255*adj[i]);
    ellipse(500,70+(100*i),10,10);
    stroke(255*x);
    line(300,70+(100*i),400,70+(100*i));
    stroke(255*x*y);
    line(400,70+(100*i),500,70+(100*i));
  }
  
  for(int i=0;i<3;i++){
    for (int j=0;j<4;j++){
      weight[i] += input[j][i] * adj[j];
    }
  }
  
  delay(100);
}




// (To-O)*(O*(1-O))
