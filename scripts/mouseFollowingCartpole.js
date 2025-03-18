class MouseFollowingCartPole {
  constructor(svgContainer, options = {}) {
    // Physics parameters
    this.options = {
      gravity: options.gravity || 9.8,
      masscart: options.masscart || 1.0,
      masspole: options.masspole || 0.1,
      length: options.length || 0.5, // actually half the pole's length
      force_mag: options.force_mag || 50.0,
      tau: options.tau || 0.02, // seconds between state updates
      kinematics_integrator: options.kinematics_integrator || 'euler',
      pole_friction: options.pole_friction || 0.1,
      
      // Visualization parameters (following cartpole.js)
      cartWidth: options.cartWidth || 60,
      cartHeight: options.cartHeight || 30,
      poleWidth: options.poleWidth || 10,
      poleHeight: options.poleHeight || 120,
      
      // Thresholds
      x_threshold: options.x_threshold || 2.4,
      theta_threshold_reward: options.theta_threshold_reward || (45 * 2 * Math.PI / 360),
      x_threshold_reward: options.x_threshold_reward || 0.5,
      
      // Number of steps to stack for observation
      stack_size: options.stack_size || 3
    };
    
    // Derived parameters
    this.options.total_mass = this.options.masspole + this.options.masscart;
    this.options.polemass_length = this.options.masspole * this.options.length;
    
    this.state = null;
    this._elapsed_steps = 0;
    
    // Initialize state history for stacked observations
    this.state_history = [];
    
    if (svgContainer) {
      this.initDrawing(svgContainer);
    }
  }
  
  initDrawing(svgContainer) {
    const { cartWidth, cartHeight, poleWidth, poleHeight } = this.options;
    
    // Get container dimensions
    this.width = svgContainer.node().getBoundingClientRect().width;
    this.height = svgContainer.node().getBoundingClientRect().height;
    this.svgContainer = svgContainer;
    
    // Set up scales to map from physics coordinates to screen coordinates
    this.xScale = d3.scaleLinear()
      .domain([-this.options.x_threshold, this.options.x_threshold])
      .range([0, this.width]);
    
    this.yScale = d3.scaleLinear()
      .domain([0, 3])
      .range([this.height, 0]);
    
    // Set background color
    this.svgContainer
      .attr("height", this.height)
      .attr("width", this.width)
      .style("background", "#DDDDDD");
    
    // Draw ground line
    this.line = this.svgContainer.append("line")
      .style("stroke", "black")
      .attr("x1", this.xScale(-this.options.x_threshold))
      .attr("y1", this.yScale(1))
      .attr("x2", this.xScale(this.options.x_threshold))
      .attr("y2", this.yScale(1));
    
    // Create cart
    this.cart = this.svgContainer.append("rect")
      .data([0])
      .attr("width", cartWidth)
      .attr("height", cartHeight)
      .attr("x", d => this.xScale(d) - cartWidth / 2)
      .attr("y", this.yScale(1) - cartHeight / 2)
      .attr("rx", 5)
      .style("fill", "BurlyWood");
    
    // Create pole group (for rotation)
    this.poleG = this.svgContainer.append("g")
      .data([180])
      .attr("transform", d => "translate(" + (this.xScale(0)) + "," + (this.yScale(1)) + ")");
    
    // Create pole
    this.pole = this.poleG.append("rect")
      .attr("x", -poleWidth / 2)
      .attr("width", poleWidth)
      .attr("height", poleHeight)
      .attr("transform", r => "rotate(" + r + ")")
      .style("fill", "SaddleBrown");
    
    // Add mouse target indicator
    this.mouseTarget = this.svgContainer.append("circle")
      .attr("r", 3)
      .style("fill", "red")
      .style("opacity", 0.8)
      .attr("cx", this.xScale(0))
      .attr("cy", this.yScale(1));
  }
  
  step(action) {
    if (!this.state) {
      console.error("Call reset before using step method.");
      return;
    }
    
    if (!(action === 0 || action === 1)) {
      console.error("Action", action, "is not valid, choose 0 for left and 1 for right.");
      return;
    }
    
    const { 
      gravity, masscart, masspole, total_mass, length, polemass_length, 
      force_mag, tau, kinematics_integrator, pole_friction, x_threshold
    } = this.options;
    
    let [x, x_dot, theta, theta_dot, mouse_x] = this.state;
    
    // Calculate force based on action
    const force = action === 1 ? force_mag : -force_mag;
    
    // Physics calculations
    const costheta = Math.cos(theta);
    const sintheta = Math.sin(theta);
    
    const temp = (force + polemass_length * Math.pow(theta_dot, 2) * sintheta) / total_mass;
    
    const thetaacc = (gravity * sintheta - costheta * temp - pole_friction * theta_dot / polemass_length) / 
                    (length * (4.0 / 3.0 - masspole * Math.pow(costheta, 2) / total_mass));
    
    const xacc = temp - polemass_length * thetaacc * costheta / total_mass;
    
    // Update state based on kinematics integrator
    if (kinematics_integrator === "euler") {
      x = x + tau * x_dot;
      x_dot = x_dot + tau * xacc;
      theta = theta + tau * theta_dot;
      theta_dot = theta_dot + tau * thetaacc;
    } else { // semi-implicit euler
      x_dot = x_dot + tau * xacc;
      x = x + tau * x_dot;
      theta_dot = theta_dot + tau * thetaacc;
      theta = theta + tau * theta_dot;
    }
    
    // Normalize theta to be between -pi and pi
    theta = ((theta + Math.PI) % (2 * Math.PI)) - Math.PI;
    
    // Update state
    this.state = [x, x_dot, theta, theta_dot, mouse_x];
    this._elapsed_steps += 1;
    
    // Update state history for stacked observations
    this.state_history.push([...this.state]);
    
    // Keep only the most recent states based on stack_size
    if (this.state_history.length > this.options.stack_size) {
      this.state_history.shift();
    }
    
    // Return stacked observation
    return {
      state: this.getStackedObservation()
    };
  }
  
  reset() {
    // Reset state to initial values
    // We'll use random small values for position and angle like in Cartpole.js
    const x = 0 + Math.random() * 0.01 - 0.005;
    const x_dot = 0;
    const theta = 0 + Math.random() * 0.01 - 0.005;
    const theta_dot = 0;
    
    // For inference, mouse_x will be set from actual mouse position
    // so we initialize it to 0 for now
    const mouse_x = 0;
    
    this.state = [x, x_dot, theta, theta_dot, mouse_x];
    this._elapsed_steps = 0;
    
    // Reset state history and initialize with copies of the initial state
    this.state_history = [];
    for (let i = 0; i < this.options.stack_size; i++) {
      this.state_history.push([...this.state]);
    }
    
    return {
      state: this.getStackedObservation()
    };
  }
  
  // Helper method to get stacked observation
  getStackedObservation() {
    // Flatten the array of state histories into a single array
    return this.state_history.flat();
  }
  
  // Method to update mouse position from actual mouse events
  updateMousePosition(mouseX) {
    if (this.state) {
      // Update the mouse position in the state
      this.state[4] = mouseX;
      
      // Update mouse target visualization
      if (this.mouseTarget) {
        this.mouseTarget
          .attr("cx", this.xScale(mouseX));
      }
    }
  }
  
  render(timestep = 20) {
    if (!this.state) return;
    
    const { cartWidth, cartHeight } = this.options;
    const [x, _, theta] = this.state;
    
    // Update pole rotation
    this.poleG.selectAll("rect")
      .transition()
      .duration(timestep)
      .attr("transform", () => "rotate(" + (theta * 180 / Math.PI + 180) + ")");
    
    // Update pole group position (follows cart)
    this.svgContainer.selectAll("g")
      .transition()
      .duration(timestep)
      .attr("transform", () => "translate(" + this.xScale(x) + "," + (this.yScale(1)) + ")");
    
    // Update cart position
    this.cart
      .transition()
      .duration(timestep)
      .attr("x", () => this.xScale(x) - cartWidth / 2);
  }
  
  // Close method (optional, for cleanup)
  close() {
    // Cleanup code here if needed
  }
} 