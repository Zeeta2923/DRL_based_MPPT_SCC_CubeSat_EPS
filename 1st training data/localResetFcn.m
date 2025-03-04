% Helper function to reset PV system conditions for the start of each simulation
function in = localResetFcn(in, mdl)
    % Set initial duty cycle to a random value
    in = in.setVariable('initialDuty', rand, 'Workspace', mdl);
    
    % Define the probability of having equal irradiation conditions
    e = 0.4;
    
    % Generate a random number to decide irradiation conditions
    random = rand;
    
    if random < e
        % If random number is less than e, select irradiation from a predefined set
        listIrradiation = 600:200:1000;
        irradiation = listIrradiation(randperm(length(listIrradiation), 1));
        Ir1 = irradiation;
    else
        % If random number is greater than or equal to e, select irradiation from another set
        listIrradiation = 100:100:600;
        irradiation = listIrradiation(randperm(length(listIrradiation), 1));
        Ir1 = irradiation;
    end
    
    % Set the variable 'Ir1' in the workspace
    in = in.setVariable('Ir1', Ir1, 'Workspace', mdl);
    
    % You can add additional parameters here if needed (e.g., temperature)
    % For example, setting a temperature variable:
    % listTemperature = 15:5:30;
    % temperature = listTemperature(randperm(length(listTemperature), 1));
    % in = in.setVariable('T1', temperature, 'Workspace', mdl);
end
