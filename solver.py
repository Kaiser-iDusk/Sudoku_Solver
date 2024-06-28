class Solver:
    N = 9
    def printing(self, arr):
        for i in range(self.N):
            for j in range(self.N):
                print(arr[i][j], end = " ")
            print()

    def isSafe(self, grid, row, col, num):
        for x in range(9):
            if grid[row][x] == num:
                return False
    
        for x in range(9):
            if grid[x][col] == num:
                return False
            
        startRow = row - row % 3
        startCol = col - col % 3
        for i in range(3):
            for j in range(3):
                if grid[i + startRow][j + startCol] == num:
                    return False
        return True

    def solve(self, grid, row, col):
        if (row == self.N - 1 and col == self.N):
            return True
        
        if col == self.N:
            row += 1
            col = 0
    
        if grid[row][col] > 0:
            return self.solve(grid, row, col + 1)
        
        for num in range(1, self.N + 1):
            if self.isSafe(grid, row, col, num):
                grid[row][col] = num
    
                if self.solve(grid, row, col + 1):
                    return True

            grid[row][col] = 0
        return False

    def make_mat(self, s):
        rows = [row for row in s.split('\n')]
        mat = []
        for row in rows:
            temp = []
            for ch in row:
                temp.append(int(ch))
            mat.append(temp)
        
        return mat

solver = Solver()