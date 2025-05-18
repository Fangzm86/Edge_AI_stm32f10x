/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    usart.c
  * @brief   This file provides code for the configuration
  *          of the USART instances.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "usart.h"

/* USER CODE BEGIN 0 */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Receive buffer
static uint8_t rxBuffer[RX_BUFFER_SIZE];
static uint16_t rxIndex = 0;
static uint8_t rxComplete = 0;

// Matrix for calculation
static Matrix receivedMatrix;

/* USER CODE END 0 */

UART_HandleTypeDef huart1;

/* USART1 init function */

void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */
  // Start receiving data in interrupt mode
  HAL_UART_Receive_IT(&huart1, &rxBuffer[rxIndex], 1);
  /* USER CODE END USART1_Init 2 */

}

void HAL_UART_MspInit(UART_HandleTypeDef* uartHandle)
{

  GPIO_InitTypeDef GPIO_InitStruct = {0};
  if(uartHandle->Instance==USART1)
  {
  /* USER CODE BEGIN USART1_MspInit 0 */

  /* USER CODE END USART1_MspInit 0 */
    /* USART1 clock enable */
    __HAL_RCC_USART1_CLK_ENABLE();

    __HAL_RCC_GPIOA_CLK_ENABLE();
    /**USART1 GPIO Configuration
    PA9     ------> USART1_TX
    PA10     ------> USART1_RX
    */
    GPIO_InitStruct.Pin = GPIO_PIN_9;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = GPIO_PIN_10;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /* USER CODE BEGIN USART1_MspInit 1 */

  /* USER CODE END USART1_MspInit 1 */
  }
}

void HAL_UART_MspDeInit(UART_HandleTypeDef* uartHandle)
{

  if(uartHandle->Instance==USART1)
  {
  /* USER CODE BEGIN USART1_MspDeInit 0 */

  /* USER CODE END USART1_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_USART1_CLK_DISABLE();

    /**USART1 GPIO Configuration
    PA9     ------> USART1_TX
    PA10     ------> USART1_RX
    */
    HAL_GPIO_DeInit(GPIOA, GPIO_PIN_9|GPIO_PIN_10);

  /* USER CODE BEGIN USART1_MspDeInit 1 */

  /* USER CODE END USART1_MspDeInit 1 */
  }
}

/* USER CODE BEGIN 1 */
/**
  * @brief  Rx Transfer completed callback
  * @param  huart: UART handle
  * @retval None
  */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
  if (huart->Instance == USART1)
  {
    // Check for buffer overflow
    if (rxIndex >= RX_BUFFER_SIZE - 1)
    {
      // Buffer full, discard oldest data
      memmove(rxBuffer, &rxBuffer[1], RX_BUFFER_SIZE - 1);
      rxIndex = RX_BUFFER_SIZE - 1;
    }
    else
    {
      rxIndex++;
    }

    // Check for end of message (newline character)
    if (rxBuffer[rxIndex - 1] == '\n')
    {
      rxBuffer[rxIndex] = '\0'; // Null-terminate the string
      rxComplete = 1;           // Mark reception as complete
      rxIndex = 0;              // Reset for next message
    }
    
    // Continue receiving with error handling
    HAL_StatusTypeDef status = HAL_UART_Receive_IT(&huart1, &rxBuffer[rxIndex], 1);
    if (status != HAL_OK)
    {
      // Error handling - attempt to restart reception
      rxIndex = 0;
      HAL_UART_Receive_IT(&huart1, &rxBuffer[rxIndex], 1);
    }
  }
}

/**
  * @brief  Process received UART data
  * @param  None
  * @retval None
  */
void UART_ProcessData(void)
{
  if (rxComplete)
  {
    // Parse the received data as a matrix
    char *token;
    char *rest = (char *)rxBuffer;
    
    // First token should be the matrix size
    token = strtok_r(rest, ",", &rest);
    if (token != NULL)
    {
      receivedMatrix.size = atoi(token);
      
      // Validate matrix size
      if (receivedMatrix.size > 0 && receivedMatrix.size <= MAX_MATRIX_SIZE)
      {
        // Parse matrix elements
        for (int i = 0; i < receivedMatrix.size; i++)
        {
          for (int j = 0; j < receivedMatrix.size; j++)
          {
            token = strtok_r(rest, ",", &rest);
            if (token != NULL)
            {
              receivedMatrix.data[i][j] = atof(token);
            }
            else
            {
              // Error in matrix format
              char errorMsg[] = "Error: Invalid matrix format\r\n";
              HAL_UART_Transmit(&huart1, (uint8_t*)errorMsg, strlen(errorMsg), HAL_MAX_DELAY);
              rxComplete = 0;
              rxIndex = 0;
              return;
            }
          }
        }
        
        // Calculate determinant
        float det = CalculateDeterminant(&receivedMatrix);
        
        // Send result back
        char resultMsg[50];
        sprintf(resultMsg, "Determinant: %.6f\r\n", det);
        HAL_UART_Transmit(&huart1, (uint8_t*)resultMsg, strlen(resultMsg), HAL_MAX_DELAY);
      }
      else
      {
        // Invalid matrix size
        char errorMsg[] = "Error: Invalid matrix size\r\n";
        HAL_UART_Transmit(&huart1, (uint8_t*)errorMsg, strlen(errorMsg), HAL_MAX_DELAY);
      }
    }
    
    // Reset for next reception
    rxComplete = 0;
    rxIndex = 0;
  }
}

/**
  * @brief  Calculate determinant of a matrix
  * @param  matrix: Pointer to the matrix structure
  * @retval Determinant value
  */
float CalculateDeterminant(Matrix* matrix)
{
  int n = matrix->size;
  
  // Base case for 1x1 matrix
  if (n == 1)
    return matrix->data[0][0];
  
  // Base case for 2x2 matrix
  if (n == 2)
    return matrix->data[0][0] * matrix->data[1][1] - matrix->data[0][1] * matrix->data[1][0];
  
  float det = 0;
  Matrix submatrix;
  submatrix.size = n - 1;
  
  // Iterate through first row elements
  for (int x = 0; x < n; x++)
  {
    // Create submatrix
    int subi = 0;
    for (int i = 1; i < n; i++)
    {
      int subj = 0;
      for (int j = 0; j < n; j++)
      {
        if (j == x)
          continue;
        
        submatrix.data[subi][subj] = matrix->data[i][j];
        subj++;
      }
      subi++;
    }
    
    // Add or subtract the determinant of the submatrix
    float sign = (x % 2 == 0) ? 1.0f : -1.0f;
    det += sign * matrix->data[0][x] * CalculateDeterminant(&submatrix);
  }
  
  return det;
}

/**
  * @brief  Transmit a float value over UART
  * @param  value: Float value to transmit
  * @retval None
  */
void UART_TransmitFloat(float value)
{
  char buffer[20];
  sprintf(buffer, "%.6f\r\n", value);
  HAL_UART_Transmit(&huart1, (uint8_t*)buffer, strlen(buffer), HAL_MAX_DELAY);
}
/* USER CODE END 1 */
